# YOLOv7-MobileNetV3 Model for Edge-Based Fire Detection

![HCMUTE Logo](https://upload.wikimedia.org/wikipedia/commons/e/e0/Logo_Đại_học_Sư_phạm_Kỹ_thuật_Thành_phố_Hồ_Chí_Minh.png)

**Author:** Doan Minh Duy Binh (Student ID: 23139005)  
**Instructor:** Assoc. Prof. Dr. Vo Minh Huan  
**Institution:** Ho Chi Minh City University of Technology and Education (HCMUTE)

## 📝 Description
Researched, trained, and optimized the YOLOv7-MobileNetV3 object detection model for the early detection of smoke, fire, and humans from aerial views. Successfully deployed the optimized model onto a Raspberry Pi 4 for standalone edge inference, delivering practical real-time performance and high accuracy.

## ✨ Features
*   **Hybrid Architecture:** Combines the robust detection capabilities of YOLOv7 with the lightweight feature extraction backbone of MobileNetV3.
*   **Edge Optimization:** Model weights are quantized to FP16 and compiled into ONNX and NCNN formats, specifically optimized for ARM architecture (Raspberry Pi 4).
*   **Real-time Performance:** Achieves up to 18.00 FPS on a Raspberry Pi 4 utilizing CPU-only inference.
*   **Grayscale Processing:** Optimized for grayscale/thermal imagery to reduce compute load and handle complex aerial environments.

## 📊 Dataset
The model was trained on a modified version of the **RGBT-3M** dataset. The data was converted to grayscale and pre-processed to remove redundant frames, resulting in 11,220 images (7,854 for training, 3,366 for validation). 
*   **Classes:** `smoke`, `fire`, `person`
*   **Dataset Link:** `[Insert your Google Drive / Kaggle link here]`

## 🚀 Performance & Results

### Accuracy (mAP)
| Class | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **All** | 0.8442 | 0.6475 | **0.7580** | 0.4332 |
| **Smoke** | 0.8784 | 0.5972 | 0.7213 | 0.4779 |
| **Fire** | 0.8620 | 0.6542 | 0.7658 | 0.4125 |
| **Person** | 0.7924 | 0.6912 | 0.7870 | 0.4093 |

### Inference Speed (Tested on Raspberry Pi 4)
| Model Format | Precision | Avg. Inference Time | FPS |
| :--- | :--- | :--- | :--- |
| PyTorch (.pt) | FP32 | 157.60 ms | 6.35 |
| ONNX (.onnx) | FP16 | 87.30 ms | 11.46 |
| **NCNN (.bin/.param)** | **FP16** | **55.55 ms** | **18.00** |

## ⚙️ Installation & Deployment

These instructions are tailored for deployment on a Raspberry Pi 4 (Headless CPU-only mode).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a Python virtual environment:**
   ```bash
   sudo apt update
   sudo apt install python3-venv python3-pip -y
   python3 -m venv drone_env
   source drone_env/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install --upgrade pip
   pip install opencv-python-headless numpy
   pip install onnxruntime ncnn
   ```

4. **Run the inference script:**
   ```bash
   python3 detect.py --model weights/best_ncnn_model --source test_video.mp4
   ```

## 📜 Acknowledgements
Special thanks to Assoc. Prof. Dr. Vo Minh Huan for his guidance throughout this project at HCMUTE.