# Wrist Bone Fracture Detection using YOLOv11

This project implements a **deep learning pipeline for wrist bone fracture detection** using the **GRAZPEDWRI-DX dataset** and the **YOLOv11 object detection architecture**. The workflow includes dataset preparation, model training, extended training, evaluation, visualization, and final result packaging.

The entire pipeline is divided into multiple modular stages to ensure clarity and reproducibility.

---

# 1. Environment Setup

The first module prepares the environment and installs the required libraries.

### Key Tasks

* Installs the **Ultralytics YOLO framework**
* Imports essential Python libraries
* Checks GPU availability for accelerated training

### Main Libraries Used

* `ultralytics` – YOLO model implementation
* `torch` – deep learning framework
* `opencv` – image processing
* `numpy` – numerical operations
* `matplotlib` – visualization
* `yaml` – configuration handling

### GPU Check

The code verifies whether CUDA is available:

```python
torch.cuda.is_available()
```

If a GPU is detected, it prints the device name, ensuring the model can utilize hardware acceleration.

---

# 2. Dataset Preparation

This module prepares the **GRAZPEDWRI-DX wrist fracture dataset** for YOLO training.

### Main Steps

1. **Define Input and Working Directories**

The dataset is read from the Kaggle input directory and reorganized inside the working directory.

```
/kaggle/working/datasets/grazpedwri
```

2. **Create YOLO-Compatible Structure**

YOLO requires a specific folder layout:

```
datasets/
 ├── images/
 │   ├── train
 │   └── val
 └── labels/
     ├── train
     └── val
```

3. **Search for Images and Labels**

The script automatically scans the dataset and collects image files:

Supported formats:

* `.jpg`
* `.png`
* `.jpeg`
* `.bmp`

Label files are located in:

```
yolov5/*.txt
```

4. **Random Dataset Splitting**

Images are randomly distributed into:

* **Training set**
* **Validation set**

This ensures proper model generalization during training.

---

# 3. High-Accuracy Model Training

The training module uses **YOLOv11s (small version)** instead of the lighter nano model.

### Why YOLOv11s?

* Higher model capacity
* Better feature extraction
* Improved detection of subtle fractures

### Training Configuration

| Parameter  | Value    |
| ---------- | -------- |
| Model      | YOLOv11s |
| Epochs     | 40       |
| Image Size | 800      |
| Batch Size | 8        |

### Medical-Specific Augmentations

Since **X-ray images are grayscale**, color augmentations are disabled:

```python
hsv_h = 0
hsv_s = 0
```

This prevents unrealistic image distortions during training.

---

# 4. Extended Training (Epoch Continuation)

To improve model performance further, training is **resumed from the last checkpoint**.

Instead of restarting training, the model continues learning using:

```
last.pt
```

### Extension Training

Initial Training:

```
40 epochs
```

Extended Training:

```
+20 epochs
```

Total Training:

```
60 epochs
```

### Implementation

```python
model = YOLO(weights_path)

model.train(
    resume=True,
    epochs=20
)
```

This allows the model to **continue learning from previous weights**, improving convergence and detection accuracy.

---

# 5. Model Visualization and Prediction

This module evaluates the trained model by running predictions on **random validation images**.

### Steps

1. Load the trained model
2. Select random images from the validation dataset
3. Run inference
4. Display detected fractures

### Visualization

Bounding boxes are drawn around detected fractures using **OpenCV and Matplotlib**.

This allows visual verification of the model’s detection capability.

---

# 6. Model Testing

The trained model is evaluated using validation images.

### Inputs

* Best trained weights:

```
best.pt
```

* Validation dataset:

```
datasets/grazpedwri/images/val
```

### Output

The system generates predictions and displays detection results for randomly selected validation images.

---

# 7. Confidence Threshold Sensitivity Analysis

Object detection models rely on a **confidence threshold** to determine whether a prediction should be accepted.

This module evaluates how different thresholds affect model performance.

### Tested Thresholds

```
0.05
0.10
0.15
0.20
0.25
0.30
0.40
0.50
```

### Metrics Evaluated

* Precision
* Recall
* F1 Score

This analysis helps determine the **optimal detection threshold** for fracture identification.

---

# 8. Test-Time Augmentation (TTA)

Test-Time Augmentation improves detection accuracy by applying transformations during inference.

### Method

For each image:

1. Apply augmentations
2. Run inference multiple times
3. Combine predictions

### Benefits

* Improved detection robustness
* Better handling of subtle fractures
* Higher evaluation metrics

### Thresholds Tested with TTA

```
0.25
0.40
0.50
0.60
```

Metrics evaluated:

* Precision
* Recall
* F1 Score
* mAP@50

---

# 9. Final Evaluation and Thesis Results

This module generates the **final evaluation metrics and visual comparisons**.

### Performance Comparison

The model results are compared against the **baseline results from a reference research paper**.

Reference:

```
Kshetri et al. (2025)
```

Metrics compared include:

* mAP
* Precision
* Recall

### Generated Visualizations

The following graphs are produced:

* Confusion Matrix
* ROC Curve
* Performance Comparison Charts

These figures are used for **thesis analysis and result reporting**.

---

# 10. Result Archiving

The final module packages all important outputs into a **submission-ready archive**.

### Files Included

* Trained model weights
* Training logs
* Evaluation graphs

Example files:

```
best.pt
results.csv
Final_Thesis_CM_ROC.png
Final_Thesis_BarChart.png
Final_Thesis_Table.png
```

### Archive Creation

The files are automatically saved inside a timestamped folder:

```
Thesis_Submission_Extended_YYYYMMDD_HHMM
```

This ensures reproducibility and proper record keeping.

---

# Project Pipeline Overview

```
Dataset Preparation
        ↓
YOLOv11 Training (40 Epochs)
        ↓
Extended Training (60 Epochs Total)
        ↓
Prediction Visualization
        ↓
Threshold Sensitivity Analysis
        ↓
Test-Time Augmentation
        ↓
Final Evaluation & Graphs
        ↓
Result Archiving
```

## 📸 Screenshots

| **Screenshot 1** | **Screenshot 2** |
| :---: | :---: |
| <img width="1764" height="829" alt="Screenshot 2026-02-17 153254" src="https://github.com/user-attachments/assets/0c842aa5-3041-434a-9bc9-5811f383feb2" /> | <img width="1618" height="796" alt="Screenshot 2026-02-17 153336" src="https://github.com/user-attachments/assets/442fd903-2539-4406-8091-84b34f0bc8ec" /> |
| **Screenshot 3** | **Screenshot 4** |
| <img width="1564" height="597" alt="Screenshot 2026-02-17 153354" src="https://github.com/user-attachments/assets/f1add549-2a19-4e34-a28b-1623d0e0f907" /> | <img width="630" height="699" alt="Screenshot 2026-02-17 153501" src="https://github.com/user-attachments/assets/ae7fe71b-d877-47ec-b92d-aa37318a5b56" /> |
| **Screenshot 5** | **Screenshot 6** |
| <img width="284" height="666" alt="Screenshot 2026-02-18 230222" src="https://github.com/user-attachments/assets/f1759d10-19d7-4db6-90f4-c73675434215" /> | <img width="859" height="758" alt="Screenshot 2026-02-17 174447" src="https://github.com/user-attachments/assets/ab49dbeb-a511-47b7-bfa4-90c364232fc8" /> |
| **Screenshot 7** | **Screenshot 8** |
| <img width="914" height="607" alt="Screenshot 2026-02-17 194529" src="https://github.com/user-attachments/assets/c751fee6-7de0-4309-b448-d8b9a8e18599" /> | <img width="426" height="379" alt="Screenshot 2026-02-17 195550" src="https://github.com/user-attachments/assets/df4f18c6-5013-429d-9cf8-acf7aeac2212" /> |
| **Screenshot 9** | **Screenshot 10** |
| <img width="594" height="433" alt="Screenshot 2026-02-18 023057" src="https://github.com/user-attachments/assets/96adc855-6d41-4230-a0cd-078d887e6292" /> | <img width="1234" height="714" alt="Screenshot 2026-02-18 025003" src="https://github.com/user-attachments/assets/51eded9c-b3da-45dc-b900-f75d23576bbf" /> |
| **Screenshot 11** | |
| <img width="1095" height="210" alt="Screenshot 2026-02-17 191524" src="https://github.com/user-attachments/assets/271adf3b-bc29-43d8-83b0-0c183cf6f197" /> | |
---

✅ **Final Model:** YOLOv11s Graz Extended
✅ **Training Duration:** 60 Epochs
✅ **Application:** Wrist Bone Fracture Detection from X-ray Images


---
