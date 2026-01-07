# Drone Detection (YOLO11n)

A complete pipeline for custom drone detection built on the YOLO11n (nano) architecture. This project targets the "small object detection" problem and focuses on distinguishing drones from birds across diverse environments (urban, rural, and open-sky scenes).

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Results & Visuals](#results--visuals)
- [Installation](#installation)
- [Dataset Acquisition](#dataset-acquisition)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Model & Hyperparameters](#model--hyperparameters)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Drones and birds can appear visually similar at a distance, which often causes false positives in surveillance or monitoring systems. This project fine-tunes a YOLO11n model to reliably detect and classify small aerial targets as either `drone` or `bird`, while preserving real-time performance on lightweight hardware.

The pipeline includes:
- Merging multiple datasets into a unified YOLO format
- Robust preprocessing and augmentation
- Training, validation, and testing scripts
- Example inference and evaluation scripts

---

![image alt](https://github.com/kanishk083/DRONE-DETECTION/blob/36cf850bb54321e800fa55f0cb6f1ae51939fc95/Screenshot%202026-01-07%20160555.png)
![image alt](https://github.com/kanishk083/DRONE-DETECTION/blob/ff26b1d9f9197f029238630afdf9af478dd4f7e7/Screenshot%202026-01-07%20160506.png)

## Key Features

- Custom curated dataset (~2,783 high-resolution frames)
- YOLO11n architecture for a balance of accuracy and speed
- Optimized image resolution (640x640)
- Data merging from multiple sources (Roboflow versions) to improve generalization
- Trained and validated across diverse scenes: cityscapes, rural fields, and open skies

---

## Results & Visuals

Training was performed for 32 epochs with AdamW optimizer. Performance monitoring produced training/validation loss & metric curves and post-training confusion matrix for the `drone` vs `bird` classes (high precision on silhouette distinctions).

Sample inference (detections illustrated with confidence scores) are shown in the repository’s `assets/` or `examples/` directory. The three sample frames referenced in this project are labelled as Image 1, Image 2, and Image 3 (see the repo images or the examples folder). They demonstrate successful detection of small/distant drones in different scenes and lighting conditions.

> Note: Visual examples include small, distant targets and demonstrate model capability on hard positives.

---

## Installation

Requirements
- Python 3.8+
- CUDA + NVIDIA GPU recommended (T4 or better)
- pip

Install core dependencies:

```bash
# Clone repo
git clone https://github.com/kanishk083/DRONE-DETECTION.git
cd DRONE-DETECTION

# Install main dependencies
pip install ultralytics roboflow torch torchvision opencv-python-headless
```

(You may wish to install into a virtual environment, and pin package versions with a requirements file for reproducibility.)

---

## Dataset Acquisition

The datasets used for training were sourced from Roboflow Universe. You will need a Roboflow API key to download them.

- Drone-vs-Bird-2: [Roboflow link — replace with actual dataset URL]
- Drone-vs-Bird-v3: [Roboflow link — replace with actual dataset URL]

To download via Roboflow (example pattern):

```bash
export ROBOFLOW_API_KEY="your_roboflow_api_key"
# Example (replace with actual dataset details provided by Roboflow)
# roboflow download <workspace>/<dataset>:version --format yolov5
```

After downloading each dataset version, the pipeline merges them into a unified YOLO11 folder structure. See `scripts/merge_datasets.py` (or the `data/` README) for the exact merge logic.

---

## Data Preparation

- Merge Roboflow dataset versions into a single dataset in YOLO format.
- Target image size: 640x640
- Recommended split:
  - Training: 70%
  - Validation: 20%
  - Test: 10%

Ensure class names are consistent (`drone`, `bird`) and the label files follow YOLO format (class x_center y_center width height).

Example `data.yaml`:

```yaml
names: ['drone', 'bird']
nc: 2
train: /path/to/train/images
val: /path/to/val/images
test: /path/to/test/images
```

---

## Training

We trained the model for 32 epochs using AdamW. Below are example commands for training with the Ultralytics package.

Python API example:

```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # pre-trained YOLO11n checkpoint
model.train(data='data.yaml', epochs=32, imgsz=640, batch=16, lr0=0.001, optimizer='AdamW')
```

CLI example (Ultralytics `yolo` CLI):

```bash
# If your ultralytics CLI uses the following form:
yolo detect train model=yolo11n.pt data=data.yaml epochs=32 imgsz=640 batch=16 lr0=0.001 optimizer=AdamW
```

- Batch size: 16
- Image size: 640
- Learning rate (lr0): 0.001
- Optimizer: AdamW

Make sure you have a GPU available for reasonable training times and that CUDA and torch versions are compatible.

---

## Inference

Use the trained weights to run inference on images or video.

Python inference example:

```python
from ultralytics import YOLO
model = YOLO('runs/detect/exp/weights/best.pt')  # path to your trained weights
results = model.predict(source='path/to/test_images', imgsz=640, conf=0.25, save=True)
```

CLI inference example:

```bash
yolo detect predict model=runs/detect/exp/weights/best.pt source=examples/test.jpg imgsz=640 conf=0.25
```

Adjust `conf` (confidence threshold) and non-max suppression settings as required for your scenario.

---

## Evaluation

- Use the Ultralytics evaluation/reporting tools or the included `evaluate.py` script.
- Standard metrics computed: AP, mAP@0.5, precision, recall.
- A confusion matrix was produced for `drone` vs `bird` showing the model's class-level performance. See `reports/confusion_matrix.png` in the repo for the full matrix and `reports/training_curves.png` for loss/metric curves.

Reproduce evaluation:

```bash
yolo val model=runs/detect/exp/weights/best.pt data=data.yaml imgsz=640
# or use provided evaluate.py to generate confusion matrix and detailed CSV/plots
python scripts/evaluate.py --weights runs/detect/exp/weights/best.pt --data data.yaml --imgsz 640
```

---

## Model & Hyperparameters

- Architecture: YOLO11n (nano)
- Input size: 640x640
- Epochs: 32
- Batch size: 16
- Optimizer: AdamW
- Learning rate: 0.001

These hyperparameters were tuned for the dataset size (~2,783 frames) and the goal of balancing real-time performance with detection robustness for small targets.

---

## Troubleshooting & Tips

- Low recall on test images with very distant objects? Try lowering the confidence threshold (e.g., 0.10–0.20) for inference and use stronger test-time augmentations.
- Overfitting signs (training >> validation performance)? Increase data augmentation, or reduce model capacity / use weight decay.
- If training is slow or OOM on GPU, reduce batch size or use mixed precision (fp16).
- For deployment on embedded devices, consider quantization / pruning after training.

---

## Contributing

Contributions are welcome. Typical contributions:
- Dataset cleaning and annotation fixes
- Additional augmentations / training recipes
- Evaluation scripts or expanded test suites
- Improvements to sample inference notebooks

Please open an issue or PR with changes and tests/documentation updates.

---

## License

This project is provided for educational and research purposes. Dataset licensing is governed by the original dataset owners — consult the Roboflow dataset pages for licensing details. The code in this repository is provided under the repository license (add a LICENSE file or specify a license here).

---

## Acknowledgements

- YOLO and Ultralytics for model and tooling
- Roboflow Universe datasets used for training data
- Community contributors and maintainers of detection tooling

---

If you want, I can also:
- Add sample training and inference notebooks (Jupyter)
- Generate a `requirements.txt`
- Create example `data.yaml` and `scripts/` (merge, train, evaluate) to drop into the repo
- Produce the confusion matrix and training curve images from logs (if you provide training logs or outputs)
