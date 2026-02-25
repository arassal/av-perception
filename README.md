# Multi-Modal Road Segmentation for Autonomous Vehicles

Real-time panoptic perception system for autonomous vehicles, developed at the **Autonomous Vehicle Laboratory** at California State Polytechnic University, Pomona.

![Segmentation Demo](images/demo_segmentation.jpg)

## Overview

This system performs simultaneous detection and segmentation of road infrastructure and dynamic objects in real time:

- **Drivable road areas** (green overlay)
- **Lane lines** (red overlay)
- **Vehicles** (bounding boxes + instance segmentation)
- **Pedestrians** (segmentation masks)
- **Bicycles** (segmentation masks)
- **Traffic lights** (detection)
- **Traffic cones** (HSV color-space + geometric detection)

The pipeline integrates **YOLOPv2** for panoptic driving perception and **YOLOv8m-seg** for instance segmentation into a unified inference system running at 20-30 FPS on GPU.

## Performance

| Task | Metric | Score |
|------|--------|-------|
| Drivable Area Segmentation | mIoU | 93.2% |
| Vehicle Detection | mAP@0.5 | 83.4% |
| Lane Detection | Accuracy | 87.3% |
| Inference Speed (V100) | FPS | 91 |

## Tech Stack

- **Deep Learning:** PyTorch, YOLOPv2, YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV, NumPy, torchvision
- **GUI:** Tkinter (desktop), Flask (web)
- **Acceleration:** CUDA, FP16 inference
- **Datasets:** BDD100K, KITTI, COCO

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script:

```bash
chmod +x setup.sh && ./setup.sh
```

### 2. Download model weights

Download the following model files and place them in the project root:

- **YOLOPv2:** Download `yolopv2.pt` and place in `data/weights/`
- **YOLOv8m-seg:** Automatically downloaded by Ultralytics on first run

### 3. Run

**Desktop GUI (recommended):**
```bash
python yolopv2_combined_gui.py
```

**Web interface:**
```bash
python web_gui.py
```

**CLI demo:**
```bash
python demo.py --source data/demo/
```

### 4. Validate setup

```bash
python validate_setup.py
```

## Project Structure

```
.
├── yolopv2_combined_gui.py     # Main desktop GUI application
├── segmentation_editor_v2.py   # Batch processing segmentation editor
├── web_gui.py                  # Flask web interface
├── demo.py                     # Command-line inference
├── validate_setup.py           # Dependency validation
├── check_camera.py             # Camera detection utility
├── download_kitti_sample.py    # KITTI dataset downloader
├── setup.sh                    # Setup script
├── requirements.txt            # Python dependencies
├── utils/
│   └── utils.py                # Core utilities (NMS, segmentation, metrics)
├── data/
│   ├── demo/                   # Demo images
│   └── weights/                # Model weights (not tracked)
├── templates/
│   └── index.html              # Web GUI template
└── images/                     # Project screenshots
```

## Features

- Real-time FPS counter and inference timing
- Adjustable confidence threshold slider
- Multi-source input: webcam, DroidCam (phone camera), video files, image galleries
- Screenshot capture with timestamps
- Lighting-invariant traffic cone detection via HSV color-space analysis
- Non-maximum suppression with configurable IoU thresholds

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- PyTorch 2.0+
- OpenCV 4.1+

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Alexander Assal** - Research Assistant, Autonomous Vehicle Laboratory
California State Polytechnic University, Pomona
[alexander@assalfamily.com](mailto:alexander@assalfamily.com) | [LinkedIn](https://www.linkedin.com/in/alexanderassal55)
