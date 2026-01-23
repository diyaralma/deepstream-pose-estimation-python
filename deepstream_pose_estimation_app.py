import sys
import numpy as np
from typing import List, Tuple
import queue as queue_module
import math
import ctypes # Pointer işlemleri için gerekli
import pyds
from post_process import refine_peaks, paf_score_graph, find_peaks, assignment, connect_parts, topology
import pyds
import gi
import platform
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
# Platform kontrolü (C++'taki #ifdef PLATFORM_TEGRA karşılığı)
# Genellikle Jetson cihazlar 'aarch64' mimarisindedir.
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
    C++ parse_objects_from_tensor_meta fonksiyonunun Python uyarlaması.
    """

    # Parametreler
    threshold = 0.1
    window_size = 5
    max_num_parts = 20
    num_integral_samples = 7
    link_threshold = 0.1
    max_num_objects = 100

    # --- Layer 0: CMAP Data Erişimi ---
    # C++: tensor_meta->out_buf_ptrs_host[0]
    layer_info_0 = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    cmap_dims = layer_info_0.inferDims

    ptr_0 = pyds.get_ptr(layer_info_0.buffer)
    # Pointer'ı float array'e cast et
    c_type_pointer_0 = ctypes.cast(ptr_0, ctypes.POINTER(ctypes.c_float))
    # NumPy array oluştur (kopyalamadan, memory view olarak)
    cmap_data = np.ctypeslib.as_array(c_type_pointer_0, shape=(cmap_dims.numElements,))

    # --- Layer 1: PAF Data Erişimi ---
    # C++: tensor_meta->out_buf_ptrs_host[1]
    layer_info_1 = pyds.get_nvds_LayerInfo(tensor_meta, 1)
    paf_dims = layer_info_1.inferDims

    ptr_1 = pyds.get_ptr(layer_info_1.buffer)
    c_type_pointer_1 = ctypes.cast(ptr_1, ctypes.POINTER(ctypes.c_float))
    paf_data = np.ctypeslib.as_array(c_type_pointer_1, shape=(paf_dims.numElements,))

    # --- Algoritma Akışı ---

    # 1. Finding peaks
    # Not: Python versiyonunda counts ve peaks return ediliyor (C++ referans ile aliyordu)
    counts, peaks = find_peaks(cmap_data, cmap_dims, threshold, window_size, max_num_parts)

    # 2. Non-Maximum Suppression
    refined_peaks = refine_peaks(counts, peaks, cmap_data, cmap_dims, window_size)

    # 3. Bipartite graph creation
    # topology global degisken olarak tanimli olmali
    score_graph = paf_score_graph(paf_data, paf_dims, topology, counts, refined_peaks, num_integral_samples)

    # 4. Assignment
    connections = assignment(score_graph, topology, counts, link_threshold, max_num_parts)

    # 5. Connect parts
    objects = connect_parts(connections, topology, counts, max_num_objects)

    return objects, refined_peaks


def create_display_meta(objects, normalized_peaks, frame_meta, frame_width, frame_height):
    """
    C++ create_display_meta fonksiyonunun Python/pyds çevirisi.
    """
    import pyds  # Fonksiyon icinde import edilebilir veya global olabilir

    K = len(topology)
    # objects listesinin uzunluğu
    # count = len(objects) # Python for dongusunde buna gerek kalmiyor ama mantik ayni

    # 1. Batch Meta ve İlk Display Meta'yı Al
    bmeta = frame_meta.base_meta.batch_meta
    dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

    for obj in objects:
        C = len(obj)

        # --- PART 1: CIRCLES (Joints / Eklemler) ---
        for j in range(C):
            k = obj[j]
            if k >= 0:
                peak = normalized_peaks[j][k]

                # Koordinat hesaplama (C++'taki gibi float * width -> int)
                # Not: C++ kodunda peak[1] X, peak[0] Y olarak kullanılmış.
                x = int(peak[1] * MUXER_OUTPUT_WIDHT)
                y = int(peak[0] * MUXER_OUTPUT_HEIGHT)

                # Meta limiti kontrolü (DeepStream'de genelde 16'dır)
                if dmeta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
                    dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
                    pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

                # Daire parametrelerini ayarla
                cparams = dmeta.circle_params[dmeta.num_circles]
                cparams.xc = x
                cparams.yc = y
                cparams.radius = 8

                # Renk: {244, 67, 54, 1}
                cparams.circle_color.red = 244
                cparams.circle_color.green = 67
                cparams.circle_color.blue = 54
                cparams.circle_color.alpha = 1.0

                cparams.has_bg_color = 1
                # Arkaplan Rengi: {0, 255, 0, 1}
                cparams.bg_color.red = 0
                cparams.bg_color.green = 255
                cparams.bg_color.blue = 0
                cparams.bg_color.alpha = 1.0

                dmeta.num_circles += 1

        # --- PART 2: LINES (Limbs / Uzuvlar) ---
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

                # Meta limiti kontrolü
                if dmeta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
                    dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
                    pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

                # Çizgi parametrelerini ayarla
                lparams = dmeta.line_params[dmeta.num_lines]
                lparams.x1 = x0
                lparams.y1 = y0
                lparams.x2 = x1
                lparams.y2 = y1
                lparams.line_width = 3

                # Renk: {0, 255, 0, 1}
                lparams.line_color.red = 0
                lparams.line_color.green = 255
                lparams.line_color.blue = 0
                lparams.line_color.alpha = 1.0

                dmeta.num_lines += 1




def pgie_src_pad_buffer_probe(pad, info, u_data):
    """
    C++ pgie_src_pad_buffer_probe fonksiyonunun Python/pyds çevirisi.
    PGIE'den gelen metadatayı ayıklar ve çizim parametrelerini günceller.
    """

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("GstBuffer alınamadı")
        return Gst.PadProbeReturn.OK

    # Batch Meta'yı buffer hash'i üzerinden alıyoruz
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    # --- Frame Meta Listesi Üzerinde Dön ---
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note: Cast işlemi C++'taki (NvDsFrameMeta *)l_frame->data ile aynıdır
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # --- 1. Frame User Meta Listesi (Frame seviyesindeki tensor verisi için) ---
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

        # --- 2. Object Meta Listesi (Obje seviyesindeki tensor verisi için) ---
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Object User Meta Listesi
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
    global frame_number  # Global sayacı içeri al

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("GstBuffer alınamadı")
        return Gst.PadProbeReturn.OK

    # Batch meta'yı buffer hash'i üzerinden al
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # --- C++'taki Object Loop ---
        # (Orijinal kodda sadece atama yapıp geçiyordu, sadık kalmak için aynısını yapıyoruz)
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Sadece cast işlemi yapılıyor, veri kullanılmıyor
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # --- Display Meta İşlemleri ---
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)

        # Text parametrelerini ayarla
        display_meta.num_labels = 1
        txt_params = display_meta.text_params[0]

        # Metni ayarla: "Frame Number = %d"
        txt_params.display_text = f"Frame Number =  {frame_number}"

        # Koordinatlar
        txt_params.x_offset = 10
        txt_params.y_offset = 12

        # Font ayarları
        txt_params.font_params.font_name = "Mono"
        txt_params.font_params.font_size = 10

        # Font Rengi (Beyaz: 1.0, 1.0, 1.0, 1.0)
        txt_params.font_params.font_color.red = 1.0
        txt_params.font_params.font_color.green = 1.0
        txt_params.font_params.font_color.blue = 1.0
        txt_params.font_params.font_color.alpha = 1.0

        # Arkaplan ayarları
        txt_params.set_bg_clr = 1
        # Arkaplan Rengi (Siyah: 0.0, 0.0, 0.0, 1.0)
        txt_params.text_bg_clr.red = 0.0
        txt_params.text_bg_clr.green = 0.0
        txt_params.text_bg_clr.blue = 0.0
        txt_params.text_bg_clr.alpha = 1.0

        # Hazırlanan display meta'yı frame'e ekle
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Frame sayacını artır
    frame_number += 1

    return Gst.PadProbeReturn.OK


def bus_call(bus, msg, loop):
    """
    C++ bus_call fonksiyonunun Python çevirisi.
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

        # Python'da g_free ve g_error_free çağrısına gerek yoktur,
        # Garbage Collector halleder.

        loop.quit()

    return True


def link_element_to_tee_src_pad(tee, sinkelem):
    """
    Tee elementinden dinamik bir pad isteyip sink elemente bağlar.
    """
    ret = False
    tee_src_pad = None
    sinkpad = None

    # C kodundaki: gst_element_request_pad(...) yerine:
    # Python'da 'get_request_pad' template'i otomatik bulur.
    tee_src_pad = tee.get_request_pad("src_%u")

    if tee_src_pad is None:
        sys.stderr.write("Failed to get src pad from tee\n")
        return False

    # Sink elementten static pad al
    sinkpad = sinkelem.get_static_pad("sink")

    if sinkpad is None:
        sys.stderr.write(f"Failed to get sink pad from '{sinkelem.get_name()}'\n")
        return False

    # Link işlemi
    if tee_src_pad.link(sinkpad) != Gst.PadLinkReturn.OK:
        sys.stderr.write(f"Failed to link '{tee.get_name()}' and '{sinkelem.get_name()}'\n")
        return False

    ret = True
    return ret


# Sabitler (C kodunda define edilmis olmali, burada tanimliyoruz)
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000


def main():
    # GStreamer Başlatma
    Gst.init(None)

    transform = None  # Sadece Tegra için

    # Input argüman kontrolü
    if len(sys.argv) != 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} <filename> <output-path>\n")
        return -1

    # Main Loop Oluşturma
    loop = GLib.MainLoop()

    # --- Element Oluşturma ---

    # Pipeline
    pipeline = Gst.Pipeline.new("deepstream-tensorrt-openpose-pipeline")

    # Source
    source = Gst.ElementFactory.make("filesrc", "file-source")

    # Parser 1 & 2
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    h264parser1 = Gst.ElementFactory.make("h264parse", "h264-parser1")

    # Decoder (GPU hızlandırmalı)
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

    # Kuyruk ve Dosya Çıkışı
    queue = Gst.ElementFactory.make("queue", "queue")
    filesink = Gst.ElementFactory.make("filesink", "filesink")

    # Çıktı dosyası yolu ayarlama
    # C: strcat(output_path,"Pose_Estimation.mp4");
    # Python'da string birleştirme daha güvenlidir.
    output_path_arg = sys.argv[2]
    if not output_path_arg.endswith("/"):
        # Basit bir path güvenliği, C kodu direkt strcat yapıyor ama biz pythonca yapalım
        pass
    final_output_path = output_path_arg + "Pose_Estimation.mp4"
    filesink.set_property("location", final_output_path)

    # Diğer Elementler
    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter1")
    tee = Gst.ElementFactory.make("tee", "TEE")
    h264encoder = Gst.ElementFactory.make("nvv4l2h264enc", "video-encoder")
    cap_filter = Gst.ElementFactory.make("capsfilter", "enc_caps_filter")

    # Caps oluşturma
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    cap_filter.set_property("caps", caps)

    qtmux = Gst.ElementFactory.make("qtmux", "muxer")

    # OSD (On Screen Display)
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")

    # Sink ve Render Elementleri
    if IS_TEGRA:
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    nvsink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink = Gst.ElementFactory.make("fpsdisplaysink", "fps-display")

    # Sink Ayarları
    # Not: C kodunda bu ayarlar yapılıyor ama sink pipeline'a eklenmiyor (commentli)
    # Ancak probe callback'ine user_data olarak 'sink' gönderildiği için oluşturmak zorundayız.
    sink.set_property("text-overlay", False)
    sink.set_property("video-sink", nvsink)
    sink.set_property("sync", False)

    # Element Kontrolü
    elements_list = [source, h264parser, decoder, pgie, nvvidconv, nvosd, sink,
                     cap_filter, tee, nvvideoconvert, h264encoder, filesink,
                     queue, qtmux, h264parser1]

    if any(e is None for e in elements_list):
        sys.stderr.write("One element could not be created. Exiting.\n")
        return -1

    if IS_TEGRA and transform is None:
        sys.stderr.write("One tegra element could not be created. Exiting.\n")
        return -1

    # Kaynak dosya ayarı
    source.set_property("location", sys.argv[1])

    # Streammux Ayarları
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

    # PGIE Ayarları
    pgie.set_property("output-tensor-meta", True)
    pgie.set_property("config-file-path", "deepstream_pose_estimation_config.txt")

    # Bus Handler
    bus = pipeline.get_bus()
    bus_watch_id = bus.add_watch(GLib.PRIORITY_DEFAULT, bus_call, loop)

    # --- Pipeline'a Element Ekleme ---
    # C++ kodundaki #ifdef ve #else bloklarına göre mantık:

    # Ortak elementler
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)

    if IS_TEGRA:
        pipeline.add(transform)
        # Tegra'da sink commentliydi, eklemiyoruz.

    # Ortak devam
    pipeline.add(tee)
    pipeline.add(nvvideoconvert)
    pipeline.add(h264encoder)
    pipeline.add(cap_filter)
    pipeline.add(filesink)
    pipeline.add(queue)
    pipeline.add(h264parser1)
    pipeline.add(qtmux)

    # --- Bağlama (Linking) ---

    # 1. Source -> Parser -> Decoder
    print("Linking Source -> Parser -> Decoder")
    source.link(h264parser)
    h264parser.link(decoder)

    # 2. Decoder -> Streammux (Request Pad ile)
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
    # C++ kodundaki aktif blok (filesink yolu):
    # stream -> pgie -> nvvidconv -> nvosd -> tee

    print("Linking Streammux chain...")
    # Tegra'da transform var mıydı? C kodunda:
    # Tegra: streammux, pgie, nvvidconv, nvosd, tee (transform araya girmiyor burada)
    # else: streammux, pgie, nvvidconv, nvosd, tee
    # Transform sadece display sink açıksa kullanılır, burada file sink var.
    # Ancak C kodunda `transform` bin'e eklendi (Tegra ise). Ama link zincirinde #if 0 içinde kalmış gibi.
    # En alttaki aktif #else bloğuna bakarsak:

    # if (!gst_element_link_many(streammux, pgie, nvvidconv, nvosd, tee, NULL))
    # Tegra bloğu için de aynısı var. Transform kullanılmıyor linklemede.

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(tee)

    # 4. Tee -> Queue -> ... -> FileSink
    # link_element_to_tee_src_pad fonksiyonunu çağırıyoruz
    if not link_element_to_tee_src_pad(tee, queue):
        sys.stderr.write("Could not link tee to queue/nvvideoconvert\n")
        return -1

    # Kuyruktan sonraki zincir
    # queue -> nvvideoconvert -> cap_filter -> h264encoder -> h264parser1 -> qtmux -> filesink
    queue.link(nvvideoconvert)
    nvvideoconvert.link(cap_filter)
    cap_filter.link(h264encoder)
    h264encoder.link(h264parser1)
    h264parser1.link(qtmux)
    qtmux.link(filesink)

    # --- Probe Ekleme ---

    # PGIE Src Pad Probe
    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stdout.write("Unable to get pgie src pad\n")
    else:
        # Not: sink pointer'ı user_data olarak geçiliyor.
        # Python'da sink objesini geçiyoruz.
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, sink)

    # OSD Sink Pad Probe
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stdout.write("Unable to get sink pad\n")
    else:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, sink)

    # --- Çalıştırma ---
    sys.stdout.write(f"Now playing: {sys.argv[1]}\n")
    pipeline.set_state(Gst.State.PLAYING)

    sys.stdout.write("Running...\n")
    try:
        loop.run()
    except:
        pass

    # --- Kapanış ---
    sys.stdout.write("Returned, stopping playback\n")
    pipeline.set_state(Gst.State.NULL)
    sys.stdout.write("Deleting pipeline\n")

    # Python'da unref işlemleri GC tarafından otomatik yapılır ama
    # GLib loop için quit çağırmak iyidir.

    return 0


if __name__ == '__main__':
    sys.exit(main())