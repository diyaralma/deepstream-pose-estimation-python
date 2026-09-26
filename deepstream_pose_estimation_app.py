import sys
import numpy as np
from typing import List, Tuple
import queue as queue_module
import math
import ctypes # Required for pointer operations
import pyds
from post_process import refine_peaks, paf_score_graph, find_peaks, assignment, connect_parts, topology
import pyds
import gi
import platform
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
# Platform check (equivalent of #ifdef PLATFORM_TEGRA in C++)
# Jetson devices typically use the 'aarch64' architecture.
IS_TEGRA = platform.machine() == 'aarch64'
# consts
EPS = 1e-6
MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDHT = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_SEC = 4000000
MAX_ELEMENTS_IN_DISPLAY_META = 16
M = 2
frame_number = 0

# Type aliases to match C++ template definitions
# Vec1D<T> equivalent
Vec1DInt = List[int]
Vec1DFloat = List[float]

# Vec2D<T> equivalent
Vec2DInt = List[List[int]]
Vec2DFloat = List[List[float]]

# Vec3D<T> equivalent
Vec3DInt = List[List[List[int]]]
Vec3DFloat = List[List[List[float]]]


def parse_objects_from_tensor_meta(tensor_meta) -> Tuple[Vec2DInt, Vec3DFloat]:
    """
    Python port of the C++ parse_objects_from_tensor_meta function.
    """

    # Parameters
    threshold = 0.1
    window_size = 5
    max_num_parts = 20
    num_integral_samples = 7
    link_threshold = 0.1
    max_num_objects = 100

    # --- Layer 0: CMAP data access ---
    # C++: tensor_meta->out_buf_ptrs_host[0]
    layer_info_0 = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    cmap_dims = layer_info_0.inferDims

    ptr_0 = pyds.get_ptr(layer_info_0.buffer)
    # Cast the pointer to a float array
    c_type_pointer_0 = ctypes.cast(ptr_0, ctypes.POINTER(ctypes.c_float))
    # Create a NumPy array (without copying, as a memory view)
    cmap_data = np.ctypeslib.as_array(c_type_pointer_0, shape=(cmap_dims.numElements,))

    # --- Layer 1: PAF data access ---
    # C++: tensor_meta->out_buf_ptrs_host[1]
    layer_info_1 = pyds.get_nvds_LayerInfo(tensor_meta, 1)
    paf_dims = layer_info_1.inferDims

    ptr_1 = pyds.get_ptr(layer_info_1.buffer)
    c_type_pointer_1 = ctypes.cast(ptr_1, ctypes.POINTER(ctypes.c_float))
    paf_data = np.ctypeslib.as_array(c_type_pointer_1, shape=(paf_dims.numElements,))

    # --- Algorithm flow ---

    # 1. Finding peaks
    # Note: the Python version returns counts and peaks (C++ took them by reference)
    counts, peaks = find_peaks(cmap_data, cmap_dims, threshold, window_size, max_num_parts)

    # 2. Non-Maximum Suppression
    refined_peaks = refine_peaks(counts, peaks, cmap_data, cmap_dims, window_size)

    # 3. Bipartite graph creation
    # topology must be defined as a global variable
    score_graph = paf_score_graph(paf_data, paf_dims, topology, counts, refined_peaks, num_integral_samples)

    # 4. Assignment
    connections = assignment(score_graph, topology, counts, link_threshold, max_num_parts)

    # 5. Connect parts
    objects = connect_parts(connections, topology, counts, max_num_objects)

    return objects, refined_peaks


def create_display_meta(objects, normalized_peaks, frame_meta, frame_width, frame_height):
    """
    Python/pyds port of the C++ create_display_meta function.
    """
    import pyds  # Can be imported inside the function or globally

    K = len(topology)
    # Length of the objects list
    # count = len(objects) # Not needed with a Python for loop, but the logic is the same

    # 1. Get the batch meta and the first display meta
    bmeta = frame_meta.base_meta.batch_meta
    dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

    for obj in objects:
        C = len(obj)

        # --- PART 1: CIRCLES (Joints) ---
        for j in range(C):
            k = obj[j]
            if k >= 0:
                peak = normalized_peaks[j][k]

                # Coordinate calculation (float * width -> int, as in C++)
                # Note: the C++ code uses peak[1] as X and peak[0] as Y.
                x = int(peak[1] * MUXER_OUTPUT_WIDHT)
                y = int(peak[0] * MUXER_OUTPUT_HEIGHT)

                # Meta limit check (usually 16 in DeepStream)
                if dmeta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
                    dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
                    pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

                # Set circle parameters
                cparams = dmeta.circle_params[dmeta.num_circles]
                cparams.xc = x
                cparams.yc = y
                cparams.radius = 8

                # Color: {244, 67, 54, 1}
                cparams.circle_color.red = 244
                cparams.circle_color.green = 67
                cparams.circle_color.blue = 54
                cparams.circle_color.alpha = 1.0

                cparams.has_bg_color = 1
                # Background color: {0, 255, 0, 1}
                cparams.bg_color.red = 0
                cparams.bg_color.green = 255
                cparams.bg_color.blue = 0
                cparams.bg_color.alpha = 1.0

                dmeta.num_circles += 1

        # --- PART 2: LINES (Limbs) ---
        for k in range(K):
            c_a = topology[k][2]
            c_b = topology[k][3]

            if obj[c_a] >= 0 and obj[c_b] >= 0:
                peak0 = normalized_peaks[c_a][obj[c_a]]
                peak1 = normalized_peaks[c_b][obj[c_b]]

                x0 = int(peak0[1] * MUXER_OUTPUT_WIDHT)
                y0 = int(peak0[0] * MUXER_OUTPUT_HEIGHT)
                x1 = int(peak1[1] * MUXER_OUTPUT_WIDHT)
                y1 = int(peak1[0] * MUXER_OUTPUT_HEIGHT)

                # Meta limit check
                if dmeta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
                    dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
                    pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

                # Set line parameters
                lparams = dmeta.line_params[dmeta.num_lines]
                lparams.x1 = x0
                lparams.y1 = y0
                lparams.x2 = x1
                lparams.y2 = y1
                lparams.line_width = 3

                # Color: {0, 255, 0, 1}
                lparams.line_color.red = 0
                lparams.line_color.green = 255
                lparams.line_color.blue = 0
                lparams.line_color.alpha = 1.0

                dmeta.num_lines += 1




def pgie_src_pad_buffer_probe(pad, info, u_data):
    """
    Python/pyds port of the C++ pgie_src_pad_buffer_probe function.
    Extracts the metadata coming from PGIE and updates the drawing parameters.
    """

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # Get the batch meta via the buffer hash
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    # --- Iterate over the frame meta list ---
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note: the cast is equivalent to (NvDsFrameMeta *)l_frame->data in C++
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # --- 1. Frame user meta list (for frame-level tensor data) ---
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            # if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
            if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                # Tensor Meta Cast
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                # tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(...)
                objects, normalized_peaks = parse_objects_from_tensor_meta(tensor_meta)

                # create_display_meta(...)
                create_display_meta(
                    objects,
                    normalized_peaks,
                    frame_meta,
                    frame_meta.source_frame_width,
                    frame_meta.source_frame_height
                )

            try:
                l_user = l_user.next
            except StopIteration:
                break

        # --- 2. Object meta list (for object-level tensor data) ---
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Object user meta list
            l_user_obj = obj_meta.obj_user_meta_list
            while l_user_obj is not None:
                try:
                    user_meta_obj = pyds.NvDsUserMeta.cast(l_user_obj.data)
                except StopIteration:
                    break

                # if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
                if user_meta_obj.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta_obj = pyds.NvDsInferTensorMeta.cast(user_meta_obj.user_meta_data)

                    objects, normalized_peaks = parse_objects_from_tensor_meta(tensor_meta_obj)

                    create_display_meta(
                        objects,
                        normalized_peaks,
                        frame_meta,
                        frame_meta.source_frame_width,
                        frame_meta.source_frame_height
                    )

                try:
                    l_user_obj = l_user_obj.next
                except StopIteration:
                    break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def osd_sink_pad_buffer_probe(pad, info, u_data):
    global frame_number  # Use the global counter

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # Get the batch meta via the buffer hash
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # --- Object loop from the C++ code ---
        # (The original code only performed the assignment; kept the same for fidelity)
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Only the cast is performed; the data is not used
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # --- Display meta operations ---
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)

        # Set text parameters
        display_meta.num_labels = 1
        txt_params = display_meta.text_params[0]

        # Set the text: "Frame Number = %d"
        txt_params.display_text = f"Frame Number =  {frame_number}"

        # Coordinates
        txt_params.x_offset = 10
        txt_params.y_offset = 12

        # Font settings
        txt_params.font_params.font_name = "Mono"
        txt_params.font_params.font_size = 10

        # Font color (white: 1.0, 1.0, 1.0, 1.0)
        txt_params.font_params.font_color.red = 1.0
        txt_params.font_params.font_color.green = 1.0
        txt_params.font_params.font_color.blue = 1.0
        txt_params.font_params.font_color.alpha = 1.0

        # Background settings
        txt_params.set_bg_clr = 1
        # Background color (black: 0.0, 0.0, 0.0, 1.0)
        txt_params.text_bg_clr.red = 0.0
        txt_params.text_bg_clr.green = 0.0
        txt_params.text_bg_clr.blue = 0.0
        txt_params.text_bg_clr.alpha = 1.0

        # Attach the prepared display meta to the frame
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Increment the frame counter
    frame_number += 1

    return Gst.PadProbeReturn.OK


def bus_call(bus, msg, loop):
    """
    Python port of the C++ bus_call function.
    """
    t = msg.type

    if t == Gst.MessageType.EOS:
        sys.stdout.write("End of Stream\n")
        loop.quit()

    elif t == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()

        # GST_OBJECT_NAME(msg->src) -> msg.src.get_name()
        # g_printerr -> sys.stderr.write
        sys.stderr.write(f"ERROR from element {msg.src.get_name()}: {err.message}\n")

        if debug:
            sys.stderr.write(f"Error details: {debug}\n")

        # In Python there is no need to call g_free and g_error_free;
        # the garbage collector handles it.

        loop.quit()

    return True


def link_element_to_tee_src_pad(tee, sinkelem):
    """
    Requests a dynamic pad from the tee element and links it to the sink element.
    """
    ret = False
    tee_src_pad = None
    sinkpad = None

    # Instead of gst_element_request_pad(...) from the C code:
    # In Python, 'get_request_pad' finds the template automatically.
    tee_src_pad = tee.get_request_pad("src_%u")

    if tee_src_pad is None:
        sys.stderr.write("Failed to get src pad from tee\n")
        return False

    # Get the static pad from the sink element
    sinkpad = sinkelem.get_static_pad("sink")

    if sinkpad is None:
        sys.stderr.write(f"Failed to get sink pad from '{sinkelem.get_name()}'\n")
        return False

    # Link
    if tee_src_pad.link(sinkpad) != Gst.PadLinkReturn.OK:
        sys.stderr.write(f"Failed to link '{tee.get_name()}' and '{sinkelem.get_name()}'\n")
        return False

    ret = True
    return ret


# Constants (presumably #define'd in the C code; defined here)
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000


def main():
    # Initialize GStreamer
    Gst.init(None)

    transform = None  # Tegra only

    # Input argument check
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <filename> <output-path>\n")
        return -1

    # Create the main loop
    loop = GLib.MainLoop()

    # --- Element creation ---

    # Pipeline
    pipeline = Gst.Pipeline.new("deepstream-tensorrt-openpose-pipeline")

    # Source
    source = Gst.ElementFactory.make("filesrc", "file-source")

    # Parser 1 & 2
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    h264parser1 = Gst.ElementFactory.make("h264parse", "h264-parser1")

    # Decoder (GPU-accelerated)
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")

    # Stream Muxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")

    if not pipeline or not streammux:
        sys.stderr.write("One element could not be created. Exiting.\n")
        return -1

    # Inference Engine (PGIE)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")

    # Converter (NV12 -> RGBA)
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")

    # Queue and file output
    queue = Gst.ElementFactory.make("queue", "queue")
    filesink = Gst.ElementFactory.make("filesink", "filesink")

    # Set the output file path
    # C: strcat(output_path,"Pose_Estimation.mp4");
    # String concatenation is safer in Python.
    output_path_arg = sys.argv[2]
    if not output_path_arg.endswith("/"):
        # Simple path safety; the C code uses strcat directly, but do it the Python way
        pass
    final_output_path = output_path_arg + "Pose_Estimation.mp4"
    filesink.set_property("location", final_output_path)

    # Other elements
    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter1")
    tee = Gst.ElementFactory.make("tee", "TEE")
    h264encoder = Gst.ElementFactory.make("nvv4l2h264enc", "video-encoder")
    cap_filter = Gst.ElementFactory.make("capsfilter", "enc_caps_filter")

    # Create caps
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    cap_filter.set_property("caps", caps)

    qtmux = Gst.ElementFactory.make("qtmux", "muxer")

    # OSD (On Screen Display)
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")

    # Sink and render elements
    if IS_TEGRA:
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    nvsink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink = Gst.ElementFactory.make("fpsdisplaysink", "fps-display")

    # Sink settings
    # Note: the C code sets these but does not add the sink to the pipeline (commented out)
    # However, it must be created because 'sink' is passed to the probe callback as user_data.
    sink.set_property("text-overlay", False)
    sink.set_property("video-sink", nvsink)
    sink.set_property("sync", False)

    # Element check
    elements_list = [source, h264parser, decoder, pgie, nvvidconv, nvosd, sink,
                     cap_filter, tee, nvvideoconvert, h264encoder, filesink,
                     queue, qtmux, h264parser1]

    if any(e is None for e in elements_list):
        sys.stderr.write("One element could not be created. Exiting.\n")
        return -1

    if IS_TEGRA and transform is None:
        sys.stderr.write("One tegra element could not be created. Exiting.\n")
        return -1

    # Source file setting
    source.set_property("location", sys.argv[1])

    # Streammux settings
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

    # PGIE settings
    pgie.set_property("output-tensor-meta", True)
    pgie.set_property("config-file-path", "deepstream_pose_estimation_config.txt")

    # Bus Handler
    bus = pipeline.get_bus()
    bus_watch_id = bus.add_watch(GLib.PRIORITY_DEFAULT, bus_call, loop)

    # --- Add elements to the pipeline ---
    # Logic follows the #ifdef and #else blocks in the C++ code:

    # Common elements
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)

    if IS_TEGRA:
        pipeline.add(transform)
        # The sink was commented out on Tegra, so it is not added.

    # Common elements (continued)
    pipeline.add(tee)
    pipeline.add(nvvideoconvert)
    pipeline.add(h264encoder)
    pipeline.add(cap_filter)
    pipeline.add(filesink)
    pipeline.add(queue)
    pipeline.add(h264parser1)
    pipeline.add(qtmux)

    # --- Linking ---

    # 1. Source -> Parser -> Decoder
    print("Linking Source -> Parser -> Decoder")
    source.link(h264parser)
    h264parser.link(decoder)

    # 2. Decoder -> Streammux (via request pad)
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write("Streammux request sink pad failed. Exiting.\n")
        return -1

    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("Decoder request src pad failed. Exiting.\n")
        return -1

    if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
        sys.stderr.write("Failed to link decoder to stream muxer. Exiting.\n")
        return -1

    # 3. Streammux -> PGIE -> NVVidConv -> NVOSD -> ...
    # Active block in the C++ code (filesink path):
    # stream -> pgie -> nvvidconv -> nvosd -> tee

    print("Linking Streammux chain...")
    # Was transform used on Tegra? In the C code:
    # Tegra: streammux, pgie, nvvidconv, nvosd, tee (transform is not in the chain here)
    # else: streammux, pgie, nvvidconv, nvosd, tee
    # Transform is only used when the display sink is enabled; here a file sink is used.
    # The C code does add `transform` to the bin (on Tegra), but in the link chain it appears to be inside #if 0.
    # Looking at the active #else block at the bottom:

    # if (!gst_element_link_many(streammux, pgie, nvvidconv, nvosd, tee, NULL))
    # The Tegra block is the same. Transform is not used in linking.

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(tee)

    # 4. Tee -> Queue -> ... -> FileSink
    # Call link_element_to_tee_src_pad
    if not link_element_to_tee_src_pad(tee, queue):
        sys.stderr.write("Could not link tee to queue/nvvideoconvert\n")
        return -1

    # Chain after the queue
    # queue -> nvvideoconvert -> cap_filter -> h264encoder -> h264parser1 -> qtmux -> filesink
    queue.link(nvvideoconvert)
    nvvideoconvert.link(cap_filter)
    cap_filter.link(h264encoder)
    h264encoder.link(h264parser1)
    h264parser1.link(qtmux)
    qtmux.link(filesink)

    # --- Add probes ---

    # PGIE Src Pad Probe
    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stdout.write("Unable to get pgie src pad\n")
    else:
        # Note: the sink pointer is passed as user_data.
        # In Python, the sink object is passed.
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, sink)

    # OSD Sink Pad Probe
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stdout.write("Unable to get sink pad\n")
    else:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, sink)

    # --- Run ---
    sys.stdout.write(f"Now playing: {sys.argv[1]}\n")
    pipeline.set_state(Gst.State.PLAYING)

    sys.stdout.write("Running...\n")
    try:
        loop.run()
    except:
        pass

    # --- Shutdown ---
    sys.stdout.write("Returned, stopping playback\n")
    pipeline.set_state(Gst.State.NULL)
    sys.stdout.write("Deleting pipeline\n")

    # In Python, unref is handled automatically by the GC, but
    # calling quit on the GLib loop is good practice.

    return 0


if __name__ == '__main__':
    sys.exit(main())