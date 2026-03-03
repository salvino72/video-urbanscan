# URBAN SCAN AI - YOLO 2026 VISION ENGINE

## 1. Project Overview & Architecture
The `analyze_video.py` script is a next-generation Python application built to analyze and audit public spaces (urban parks, gardens, and pathways) through video feeds. 
Rather than relying on older algorithms, this engine leverages **Ultralytics YOLO (World-v2 Large)** as an "Open-Vocabulary" backend, enabling the dynamic tracking of up to 40 intricate objects simultaneously via natural language queries, without needing a dedicated training dataset for every single item.

## 2. Advanced AI Object Mapping
We categorize the detected objects into several highly specific clusters:
*   **CRITICAL_DAMAGE:** Monitors structural decay such as `deep pavement pothole`, `broken concrete curb edge`, or `cracked paving stones` (Highlighted in Bright Red).
*   **DIRTY_PAVEMENT:** Tracks environmental dirt like `mud puddle on path` and `loose sand accumulation`.
*   **LITTER & WASTE:** Identifies garbage instances including `plastic bottle trash`, `cigarette butt`, and `rusty can` (Highlighted in Orange).
*   **PUBLIC_ASSETS & VEGETATION_STATE:** Categorizes environmental strengths, from `wooden gazebo` to `blooming flower beds`.

## 3. Real-Time Hardware Optimization Engine
The application is built with incredible scalability in mind, able to dynamically switch its compute pipeline based on the target architecture:

### A. Extreme NVIDIA GPU Acceleration (CUDA)
By default, executing the script utilizes PyTorch to interface with Nvidia Tensor Cores. 
**Optimization Strategy:** The code dynamically forces the tensor input resolution (`imgsz`) to a massive **1920x1080 (Native Full HD)**, activating the `half=True` (FP16 half-precision format) and `augment=True` parameters. This pushes the RTX architecture to near peak-load without bottlenecks, maximizing graphical fidelity over purely raw millisecond speed.

### B. Intel Engine & OpenVINO (CPU / ARC)
When bypassing Nvidia (`--device cpu` / `--device arc`), the script automatically engages its integrated fallback handling. 
It requires the user to pre-export the `.pt` models to Intel's highly efficient OpenVINO format. Once detected, the model is routed through the OpenVINO framework, automatically scaling the internal scan resolution down to `imgsz=832` to guarantee smooth, continuous execution threads on CPU-only or Arc GPU machines. 

## 4. Privacy-First Human Blurring
A key highlight of this script is the built-in algorithmic **Privacy Blur**.
When people are detected in the frame:
1. The script isolates all human boundary boxes.
2. It calculates the mathematical area (width x height) of each box to identify the closest, largest person. This person is assumed to be the "Host/Subject" and is authorized (`PERSON CONFIRMED`).
3. For every other person in the background, the system intercepts the upper 1/3 (head level) of their bounding box coordinates.
4. It isolates that precise sector, applies a severe **Gaussian Blur (51, 51, 30)** matrix, and draws a shaded circular overlay to effectively hide facial features (`PRIVACY BLUR`).

## 5. Evolved Analytics HUD
Instead of just printing bounding boxes, OpenCV generates a complete Heads-Up Display:
*   **Park Health Score**: Actively compares positive assets (vegetation, sculptures) against negative assets (potholes, garbage) to assign a dynamic 0-100% health grade to the environment.
*   **Live Inference Data & Logging**: Shows chronological lists of identified elements and calculates frame-time precision, keeping users perpetually informed of both backend constraints and frontend discoveries.
