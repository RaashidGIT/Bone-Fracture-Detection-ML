# Bone Fracture Classification using CNN with Attention

This project implements a **deep learning pipeline for binary bone fracture classification** using grayscale X-ray images. The system performs **duplicate removal, edge-aware preprocessing, CNN training with focal loss, threshold optimization, and final deployment packaging**.

The model is designed to **prioritize fracture detection recall**, which is critical in medical imaging where missing a fracture (false negative) can have serious consequences.

---

# 1. Environment Setup

The first module imports all required libraries and verifies that **TensorFlow and GPU acceleration** are available.

### Key Libraries

| Library            | Purpose                       |
| ------------------ | ----------------------------- |
| TensorFlow / Keras | Deep learning model training  |
| OpenCV             | Image processing              |
| NumPy              | Numerical computation         |
| scikit-learn       | Data splitting and evaluation |
| tqdm               | Progress bars                 |
| hashlib            | Duplicate image detection     |

Example initialization:

```python
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU:", len(tf.config.list_physical_devices("GPU")) > 0)
```

This ensures the training environment is correctly configured.

---

# 2. Training Configuration

All important hyperparameters are defined inside a configuration dictionary.

### Training Parameters

| Parameter     | Value     |
| ------------- | --------- |
| Image Size    | 224 × 224 |
| Batch Size    | 16        |
| Epochs        | 120       |
| Learning Rate | 3e-4      |
| Test Split    | 20%       |
| Random Seed   | 42        |

Example configuration:

```python
CONFIG = {
    "img_size": (224,224),
    "batch_size": 16,
    "epochs": 120,
    "learning_rate": 3e-4
}
```

Using a centralized configuration improves **experiment reproducibility**.

---

# 3. Dataset Loading

The dataset used is the **Ultimate Mixed Bone Fracture Dataset**, organized into two classes:

* `fractured`
* `non_fractured`

The script automatically verifies that the dataset path exists before training begins.

```python
DATA_DIR = "/kaggle/input/bone-fracture-dataset/.../images"
```

Each class folder contains the corresponding X-ray images.

---

# 4. Duplicate Image Detection

Medical datasets often contain **duplicate or repeated images**, which can artificially inflate model accuracy.

To prevent this, the pipeline detects **pixel-identical duplicates** using **MD5 hashing**.

### Duplicate Detection Process

1. Each image file is read.
2. A **hash signature** is generated.
3. Files with identical hashes are marked as duplicates.

Example hashing function:

```python
def file_md5(path):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        md5.update(f.read())
    return md5.hexdigest()
```

Duplicate images are removed before training to ensure **dataset integrity**.

---

# 5. Edge-Aware Image Preprocessing

X-ray images require specialized preprocessing to enhance **bone edges and fracture lines**.

### Preprocessing Steps

1. Convert image to **grayscale**
2. Resize image to **224 × 224**
3. Apply **CLAHE contrast enhancement**
4. Extract **edge information using Sobel filtering**
5. Combine edges with the original image
6. Normalize pixel values

Example concept:

```python
edges = cv2.Sobel(img, cv2.CV_32F, 1, 1)
img = 0.8 * img + 0.2 * edges
```

This technique improves the visibility of **fine fracture structures**.

---

# 6. Dataset Creation

After preprocessing:

* Images are stored in `X`
* Labels are stored in `y`

Label encoding:

| Label | Meaning       |
| ----- | ------------- |
| 0     | Non-fractured |
| 1     | Fractured     |

The dataset is converted into NumPy arrays for efficient training.

---

# 7. Train-Test Split

The dataset is split using **stratified sampling** to maintain class balance.

```python
train_test_split(
    X, y,
    test_size=0.2,
    stratify=y
)
```

This ensures that both fracture and non-fracture samples appear proportionally in training and testing sets.

---

# 8. Class Weighting (False Negative Optimization)

In medical diagnosis, **false negatives are more dangerous than false positives**.

To address this, the pipeline assigns **higher weight to fractured images** during training.

Example:

```python
class_weight = {
    0: weight_non_fracture,
    1: weight_fracture
}
```

This forces the model to **prioritize fracture detection**.

---

# 9. Data Augmentation

To improve model generalization, the training images undergo **controlled augmentation**.

### Augmentations Used

* Random rotation
* Random translation
* Random zoom
* Random contrast

Example:

```python
layers.RandomRotation(0.05)
layers.RandomZoom(0.08)
```

These transformations simulate **variations in X-ray imaging conditions**.

---

# 10. CNN Architecture with Attention

The model is a **custom convolutional neural network** with three convolutional blocks.

### Convolution Block Structure

Each block contains:

* Convolution layer
* Batch normalization
* ReLU activation
* Max pooling
* Dropout

Example block:

```python
Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool
```

### Attention Mechanism

An **attention map** is generated to focus on important regions.

```python
att = Conv2D(1,1, activation="sigmoid")(x)
x = layers.multiply([x, att])
```

This allows the model to emphasize **fracture-relevant features**.

---

# 11. Focal Loss for Medical Classification

Standard cross-entropy loss struggles with **class imbalance**.

This project uses **Focal Loss**, which focuses training on difficult samples.

```python
focal_loss(gamma=2.5, alpha=0.75)
```

Benefits:

* Improves minority class learning
* Reduces dominance of easy samples
* Improves fracture detection performance

---

# 12. Model Training

The model is trained using:

| Component               | Setting |
| ----------------------- | ------- |
| Optimizer               | AdamW   |
| Epochs                  | 120     |
| Batch Size              | 16      |
| Early Stopping          | Enabled |
| Learning Rate Scheduler | Enabled |

Callbacks used:

* **EarlyStopping** – prevents overfitting
* **ReduceLROnPlateau** – adjusts learning rate dynamically

---

# 13. Threshold Optimization

Instead of using a fixed **0.5 classification threshold**, the system evaluates multiple thresholds.

Tested thresholds:

```
0.30
0.35
0.40
0.45
0.50
0.55
0.60
```

For each threshold, the model calculates:

* Accuracy
* Recall
* False Positives
* False Negatives

This helps identify the **best decision boundary**.

---

# 14. Best Threshold Selection

The best threshold is selected using two criteria:

1. **Maximum accuracy**
2. **Recall greater than 0.90**

This ensures high fracture detection sensitivity.

---

# 15. Final Model Evaluation

The final model is evaluated using:

* Confusion Matrix
* Classification Report
* Accuracy
* Recall (Sensitivity)
* Specificity

Example metrics:

```
Accuracy
Recall (Fracture detection)
Specificity (Healthy detection)
```

These metrics provide a comprehensive evaluation of model performance.

---

# 16. Model Export for Deployment

The trained model is saved in **Keras format** for deployment.

```python
model.save("fracture_model_224_final.keras")
```

A deployment configuration file is also created.

Example:

```json
{
  "best_threshold": 0.45,
  "accuracy": 0.94,
  "recall": 0.92,
  "specificity": 0.96
}
```

This allows the model to be easily integrated into **web applications or diagnostic systems**.

---

# Complete Pipeline Overview

```
Dataset Loading
       ↓
Duplicate Removal
       ↓
Edge-Aware Preprocessing
       ↓
Train/Test Split
       ↓
Data Augmentation
       ↓
CNN + Attention Model
       ↓
Focal Loss Training
       ↓
Threshold Optimization
       ↓
Final Evaluation
       ↓
Model Export
```

---

✅ **Model Type:** CNN with Attention
✅ **Input:** 224×224 grayscale X-ray images
✅ **Output:** Fracture / Non-fracture classification
✅ **Framework:** TensorFlow / Keras

---
