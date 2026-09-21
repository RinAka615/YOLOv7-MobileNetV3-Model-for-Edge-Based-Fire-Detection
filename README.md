# YOLOv7-MobileNetV3-Model-for-Edge-Based-Fire-Detection
An edge AI system utilizing YOLOv7-MobileNetV3 to detect smoke, fire, and humans from aerial imagery, deployed on a Raspberry Pi 4 for standalone inference.

## 📌 Project Overview
This project focuses on researching, training, and optimizing a hybrid object detection model for early fire warning systems using UAVs (Drones). By replacing the original YOLOv7 backbone with **MobileNetV3**, the model is extremely lightweight and optimized for edge devices with limited computing resources (pure CPU inference).

## ✨ Key Features
- **Hybrid Architecture:** YOLOv7 combined with MobileNetV3 backbone for efficient feature extraction with minimal parameters.
- **Edge Deployment:** Successfully compiled and quantized to **NCNN (FP16)**, enabling standalone inference on ARM-based processors without a dedicated GPU.
- **Target Classes:** `smoke`, `fire`, `person`.
- **Grayscale Optimization:** The model is trained on grayscale images to reduce noise from lighting variations and minimize computational load.

## 📊 Model Specifications & Performance
The model achieves an optimal balance between accuracy and inference speed on edge devices:
- **Parameters:** 332,253
- **Layers:** 372
- **Compute:** 1.4 GFLOPs

### Accuracy (mAP@50)
| Class | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **All** | 0.8442 | 0.6475 | **75.80%** | 43.32% |
| **Person** | 0.7924 | 0.6912 | **78.70%** | 40.93% |
| **Fire** | 0.8620 | 0.6542 | **76.58%** | 41.25% |
| **Smoke** | 0.8784 | 0.5972 | **72.13%** | 47.79% |

### Edge Inference Speed (Tested on Raspberry Pi 4)
| Format | Precision | Inference Time | FPS |
| :--- | :--- | :--- | :--- |
| PyTorch (`.pt`) | FP32 | 157.60 ms | 6.35 FPS |
| ONNX (`.onnx`) | FP16 | 87.30 ms | 11.46 FPS |
| **NCNN (`.bin`, `.param`)** | **FP16** | **55.55 ms** | **18.00 FPS** |

## 🗂️ Dataset
The model was trained on a modified version of the **RGBT-3M** dataset.
- Total training/validation frames: **11,220 images** (Grayscale).
- Labels: 13,574 (smoke), 11,315 (fire), 5,888 (person).
- **Dataset Link:** [https://zenodo.org/records/13732947](url)

## ⚙️ Installation & Deployment

These instructions are tailored for deployment on a Raspberry Pi 4 (Headless CPU-only mode).

**1. Clone the repository:**
```bash
git clone [https://github.com/RinAka615/YOLOv7-MobileNetV3-Model-for-Edge-Based-Fire-Detection.git](https://github.com/RinAka615/YOLOv7-MobileNetV3-Model-for-Edge-Based-Fire-Detection.git)
cd YOLOv7-MobileNetV3-Model-for-Edge-Based-Fire-Detection
```

**2. Create a Python virtual environment:**
```bash
sudo apt update
sudo apt install python3-venv python3-pip -y
python3 -m venv drone_env
source drone_env/bin/activate
```

**3. Install required dependencies:**

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Core image processing and mathematical operations
pip install opencv-python-headless numpy Pillow scipy

# YOLOv7 utility libraries (configs, progress bars, data handling)
pip install PyYAML tqdm requests pandas matplotlib seaborn

# Edge inference engines
pip install onnxruntime ncnn

# CPU-only PyTorch (Required if the inference script imports torch/torchvision)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```
**Lưu ý:** Lệnh cài đặt `torch` và `torchvision` được chỉ định tải bản dành riêng cho CPU (`--index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)`) để tối ưu hóa dung lượng và tránh lỗi khi cài trên vi tính nhúng như Raspberry Pi. Nếu file nhận diện (`detect.py` / `inference.py`) được viết lại thuần túy chỉ dùng `ncnn` hoặc `onnxruntime` mà không gọi thư viện `torch`, có thể bỏ dòng cuối cùng đi để Pi chạy nhẹ hơn nhé.

**4. Run the inference script:**
```bash
python3 detect.py --model weights/best_ncnn_model --source test_video.mp4
```
## 📜 Acknowledgements
Special thanks to Assoc. Prof. Dr. Vo Minh Huan for his guidance throughout this project at HCM-UTE.

