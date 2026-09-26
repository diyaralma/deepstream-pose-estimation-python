#!/usr/bin/env python3
import sys
import gi
import ctypes
import math
import numpy as np
import post_process
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
PROCESS_INTERVAL = 1
MAX_LIMB_DISTANCE = VIDEO_WIDTH * 0.40

# Body-25 skeleton structure
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12),
    (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18)
]


def parse_body_pose_smooth(heatmap_layer, layer_dims):
    c, h, w = layer_dims[:3]
    try:
        data = heatmap_layer.reshape(c, h, w)
    except:
        try:
            c, h, w = layer_dims[1:4]
            data = heatmap_layer.reshape(c, h, w)
        except:
            return {}
    threshold = 0.1
    detected_body_parts = {}
    for part_idx in range(c):
        part_map = data[part_idx, :, :]
        max_val = np.max(part_map)
        if max_val > threshold:
            pos = np.unravel_index(np.argmax(part_map), (h, w))
            grid_y, grid_x = pos[0], pos[1]
            if 1 < grid_x < w - 1 and 1 < grid_y < h - 1:
                val_x_minus = part_map[grid_y, grid_x - 1]
                val_x_plus = part_map[grid_y, grid_x + 1]
                val_y_minus = part_map[grid_y - 1, grid_x]
                val_y_plus = part_map[grid_y + 1, grid_x]
                offset_x = 0.25 if val_x_plus > val_x_minus else -0.25 if val_x_plus < val_x_minus else 0
                offset_y = 0.25 if val_y_plus > val_y_minus else -0.25 if val_y_plus < val_y_minus else 0
                refined_x = grid_x + offset_x
                refined_y = grid_y + offset_y
            else:
                refined_x, refined_y = grid_x, grid_y
            scale_x = VIDEO_WIDTH / w
            scale_y = VIDEO_HEIGHT / h
            real_x = int(refined_x * scale_x)
            real_y = int(refined_y * scale_y)
            detected_body_parts[part_idx] = (real_x, real_y)
    return detected_body_parts


# ---- Bus Call (error handling) ----
def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("\nVideo finished (End of Stream)\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("ERROR: %s: %s\n" % (err, debug))
        loop.quit()
    return True


# ---- Tensor probe ----
def pgie_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        if frame_meta.frame_num % PROCESS_INTERVAL != 0:
            l_frame = l_frame.next
            continue
        l_user = frame_meta.frame_user_meta_list
        heatmap_data = None
        heatmap_dims = None
        while l_user is not None:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                for i in range(tensor_meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                    dims = layer.dims.d[:layer.dims.numDims]
                    if dims[0] < 30:
                        ptr = pyds.get_ptr(layer.buffer)
                        c_type_pointer = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
                        num_elements = np.prod(dims)
                        heatmap_data = np.ctypeslib.as_array(c_type_pointer, shape=(num_elements,)).copy()
                        heatmap_dims = dims
                        break
            l_user = l_user.next
        if heatmap_data is not None:
            results = parse_body_pose_smooth(heatmap_data, heatmap_dims)
            if results:
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                valid_lines = []
                for (idx_a, idx_b) in POSE_PAIRS:
                    if idx_a in results and idx_b in results:
                        pt_a = results[idx_a]
                        pt_b = results[idx_b]
                        dist = math.sqrt((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2)
                        if dist < MAX_LIMB_DISTANCE:
                            valid_lines.append((pt_a, pt_b))
                display_meta.num_lines = len(valid_lines)
                for l_idx, (pt1, pt2) in enumerate(valid_lines):
                    l_params = display_meta.line_params[l_idx]
                    l_params.x1, l_params.y1, l_params.x2, l_params.y2 = pt1[0], pt1[1], pt2[0], pt2[1]
                    l_params.line_width = 3
                    l_params.line_color.set(0.0, 1.0, 0.0, 1.0)
                display_meta.num_circles = len(results)
                for c_idx, (_, (x, y)) in enumerate(results.items()):
                    c_params = display_meta.circle_params[c_idx]
                    c_params.xc, c_params.yc = x, y
                    c_params.radius = 4
                    c_params.circle_color.set(1.0, 0.0, 0.0, 1.0)
                    c_params.has_bg_color = 1
                    c_params.bg_color.set(1.0, 0.0, 0.0, 1.0)
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


# ---- NEW: Decodebin callback ----
# Catches the video output of the automatic decoder and links it to streammux
def cb_decodebin_newpad(decodebin, pad, streammux):
    caps = pad.get_current_caps()
    if not caps: return
    name = caps.to_string()
    # If the incoming data is video
    if "video" in name:
        sinkpad = streammux.get_request_pad("sink_0")
        if not sinkpad.is_linked():
            # Try to link
            res = pad.link(sinkpad)
            if res == Gst.PadLinkReturn.OK:
                print("Decodebin successfully linked to Streammux.")
            else:
                print(f"Decodebin link error: {res}")


def main(args):
    if len(args) < 2:
        print("Usage: python3 main_pose.py <video.mp4>")
        return
    input_file = args[1]
    if input_file.startswith("file://"):
        input_file = input_file[7:]

    Gst.init(None)
    pipeline = Gst.Pipeline()

    # --- ELEMENTS ---
    source = Gst.ElementFactory.make("filesrc", "file-source")

    # CHANGE: a single 'decodebin' instead of qtdemux, parser and decoder
    decodebin = Gst.ElementFactory.make("decodebin", "decode-bin")

    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nference-engine")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
    nvosd = Gst.ElementFactory.make("nvdsosd", "nvosd")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    if not all([source, decodebin, streammux, pgie, nvvidconv, nvosd, sink]):
        print("Failed to create elements")
        return

    # --- SETTINGS ---
    source.set_property("location", input_file)
    streammux.set_property("width", VIDEO_WIDTH)
    streammux.set_property("height", VIDEO_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("live-source", 0)
    pgie.set_property("config-file-path", "deepstream_pose_estimation_config.txt")
    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    # --- ADD TO PIPELINE ---
    # Note: h264parse and nvv4l2decoder are no longer used; decodebin does their job
    for e in [source, decodebin, streammux, pgie, nvvidconv, nvosd, sink]:
        pipeline.add(e)

    # --- LINKS ---
    source.link(decodebin)

    # Decodebin is a dynamic element; it is linked once its pad is created:
    decodebin.connect("pad-added", cb_decodebin_newpad, streammux)

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # --- PROBE ---
    pgie_src_pad = pgie.get_static_pad("src")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    # --- LOOP AND ERROR MONITORING ---
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        print(f"Pipeline running: {input_file}")
        loop.run()
    except Exception as e:
        print(e)
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main(sys.argv)