# DeepStream Human Pose Estimation in Python

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![DeepStream](https://img.shields.io/badge/NVIDIA%20DeepStream-6.x%20%7C%207.x-76B900?logo=nvidia&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-FP16-76B900?logo=nvidia&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Jetson%20%7C%20dGPU-lightgrey)

Multi-person human pose estimation on video with **NVIDIA DeepStream**. This is a **Python port** of NVIDIA's C++ reference application, [NVIDIA-AI-IOT/deepstream_pose_estimation](https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation).

The model runs as a TensorRT engine inside `nvinfer`. Its raw output tensors are read in a pad probe and decoded into skeletons in Python. The skeletons are drawn with DeepStream display metadata, shown on screen and saved to MP4.

> For the live, multi-stream (RTSP) version of this pipeline, see [pose-estimation-dynamic-source-handler](https://github.com/diyaralma/pose-estimation-dynamic-source-handler).

## How it works

```
filesrc ─► h264parse ─► nvv4l2decoder ─► nvstreammux ─► nvinfer (pose model, FP16)
        ─► nvvideoconvert ─► nvdsosd ─► tee ─┬─► display
                                              └─► nvv4l2h264enc ─► qtmux ─► Pose_Estimation.mp4
```

1. `nvinfer` runs with `network-type=100` and `output-tensor-meta=1`, so DeepStream skips its built-in parsing and attaches the raw output tensors (part confidence maps and part affinity fields) to each frame.
2. A probe on the `nvinfer` source pad reads those tensors through `pyds` and runs the post-processing pipeline, ported from the C++ original:
   - Peak finding and sub-pixel refinement on the confidence maps
   - Part-affinity-field scoring between candidate joints
   - Optimal limb assignment with the **Munkres (Hungarian) algorithm**
   - Connected-component grouping into individual people
3. Keypoints and limbs are drawn with DeepStream display metadata: circles for joints, lines for limbs.

## Files

| File | Purpose |
|---|---|
| [`deepstream_pose_estimation_app.py`](deepstream_pose_estimation_app.py) | Main application: pipeline, tensor probe, drawing, MP4 output |
| [`post_process.py`](post_process.py) | Peak detection, PAF scoring, part connection (port of the C++ post-processing) |
| [`munkres_algorithm.py`](munkres_algorithm.py), [`pair_graph.py`](pair_graph.py), [`cover_table.py`](cover_table.py) | Hungarian algorithm used for limb assignment |
| [`main.py`](main.py) | Simplified experimental variant with a lightweight per-joint heatmap parser |
| [`deepstream_pose_estimation_config.txt`](deepstream_pose_estimation_config.txt) | `nvinfer` configuration (FP16, raw tensor output) |
| `common/` | Platform detection, FPS measurement, bus handling |

## Getting started

### Requirements

- NVIDIA DeepStream SDK 6.x / 7.x with the [Python bindings (`pyds`)](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
- NVIDIA dGPU or Jetson
- Python 3 with `numpy`

### Run

```bash
git clone https://github.com/diyaralma/pose-estimation.git
cd pose-estimation

# <input.h264> is a raw H.264 elementary stream; the output is written to <output-dir>/Pose_Estimation.mp4
python3 deepstream_pose_estimation_app.py <input.h264> <output-dir>/
```

On the first run DeepStream builds a TensorRT engine from `pose_estimation.onnx` for your GPU, which takes a few minutes. The engine is cached for later runs.

## Acknowledgements

- [NVIDIA-AI-IOT/deepstream_pose_estimation](https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation): the original C++ application, post-processing logic and model (MIT License)
- [NVIDIA-AI-IOT/trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose): the pose estimation network
- [NVIDIA DeepStream Python Apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
