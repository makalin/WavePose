# WavePose

WavePose is an open-source implementation of "DensePose From WiFi" — a deep learning framework that uses WiFi signals (Channel State Information) to reconstruct dense 2D/3D human pose maps.  
No cameras, no LiDAR, no privacy concerns — just off-the-shelf WiFi routers and antennas.

---

## 📜 Overview
WavePose maps WiFi Channel State Information (CSI) to DensePose UV coordinates, enabling robust human pose estimation in low-light, occluded, or privacy-sensitive environments.

It builds on the methods described in the [DensePose From WiFi](https://arxiv.org/pdf/2301.00250.pdf) paper, integrating:
- **Phase Sanitization** to stabilize noisy CSI phase data
- **Modality Translation Network** to convert 1D WiFi signals into image-like features
- **WiFi-DensePose RCNN** for multi-person dense pose estimation
- **Keypoint Supervision & Transfer Learning** for faster convergence and better accuracy

---

## 🚀 Features
- **Privacy-Preserving** — No cameras required  
- **Low-Cost Hardware** — Works with $30 WiFi routers  
- **Multi-Person Tracking** — Handles multiple people in view  
- **Lighting & Occlusion Resistant** — Works in the dark or behind obstacles  
- **DensePose-Level Detail** — UV mapping with up to 24 body regions

---

## 🛠 Installation

```bash
git clone https://github.com/makalin/WavePose.git
cd WavePose
pip install -r requirements.txt
````

**Requirements:**

* Python 3.10+
* PyTorch ≥ 1.12.0
* Detectron2 ≥ 0.6
* NumPy, SciPy, OpenCV, Matplotlib

---

## 📂 Dataset

WavePose uses synchronized CSI and video datasets.
Download the "Person-in-WiFi" dataset from [ICCV 2019](https://arxiv.org/abs/1904.00276) and extract into the `data/` folder.

---

## ⚙️ Usage

### 1. Train from Scratch

```bash
python train.py --data_dir ./data --epochs 50 --batch_size 16
```

### 2. Inference

```bash
python infer.py --csi_file sample_csi.npy --output_dir ./results
```

---

## 📊 Model Architecture

1. **CSI Preprocessing**

   * Amplitude & phase extraction
   * Phase unwrapping & sanitization
2. **Modality Translation Network**

   * MLP-based encoding of amplitude & phase tensors
   * 1D→2D reshaping & convolutional upsampling
3. **WiFi-DensePose RCNN**

   * ResNet-FPN backbone
   * Keypoint & DensePose heads
4. **Transfer Learning**

   * Teacher-student setup using pretrained image-based DensePose

---

## 📈 Performance

| Metric      | AP\@50 | dpAP·GPSm\@50 |
| ----------- | ------ | ------------- |
| WiFi-Based  | 87.2   | 77.4          |
| Image-Based | 94.4   | 94.9          |

---

## 🧪 Future Work

* Multi-layout training for domain generalization
* 3D body shape reconstruction from WiFi
* Real-time inference optimizations

---

## 📄 Citation

If you use this code, please cite:

```
@article{geng2023denseposewifi,
  title={DensePose From WiFi},
  author={Geng, Jiaqi and Huang, Dong and De la Torre, Fernando},
  journal={arXiv preprint arXiv:2301.00250},
  year={2023}
}
```

---

## 📜 License

MIT License. See `LICENSE` for details.
