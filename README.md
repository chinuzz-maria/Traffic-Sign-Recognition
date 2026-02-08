# Traffic-Sign-Recognition
Traffic sign recognition using data mining techniques

# Traffic Sign Recognition Using Data Mining Techniques

A comprehensive traffic sign recognition system developed as part of the Data Mining course at BITS Pilani Dubai Campus. This project implements multiple classification and clustering algorithms to accurately identify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [References](#references)

---

## 🎯 Project Overview

This project focuses on traffic sign recognition in complex outdoor environments where variable lighting, partial occlusions, and unpredictable positions pose significant challenges. The system employs:

- **Preprocessing** to enhance image quality and reduce noise
- **Classification algorithms** to accurately identify traffic sign types
- **Clustering techniques** to group similar features and analyze data structure
- **Evaluation metrics** to assess model performance comprehensively

**Keywords:** Traffic Sign Recognition, Intelligent Vehicle, Driver Assistance Systems, Machine Learning, Deep Learning

---

## 📊 Dataset

### German Traffic Sign Recognition Benchmark (GTSRB)

- **Total Images:** ~50,000 images
- **Classes:** 43 traffic sign categories
- **Training Set:** ~39,000 images (31,367 after preprocessing)
- **Test Set:** ~12,000 images (7,842 after preprocessing)
- **Format:** Color images (RGB), JPEG format
- **Dimensions:** Resized to 64×64 pixels for processing

**Dataset Features:**
- Real-world scenarios with varying lighting conditions
- Different weather effects
- Partial occlusions
- Multiple viewing angles

---

## 🛠️ Installation

### Prerequisites

```bash
Python 3.10+
Google Colab (recommended) or local environment with GPU support
```

### Required Libraries

```bash
pip install tensorflow
pip install scikit-learn
pip install opencv-python
pip install numpy pandas
pip install matplotlib seaborn
pip install tqdm
```

### Hardware Requirements

**Recommended (Google Colab):**
- GPU: NVIDIA Tesla T4 (16 GB VRAM) or Tesla P100
- CPU: 2-core virtual CPU
- RAM: 12 GB
- Storage: ~70 GB temporary disk

**Minimum (Local):**
- Processor: Intel Core i7 (12th Gen) @ 2.10 GHz
- RAM: 16 GB DDR4
- GPU: NVIDIA GeForce RTX 3050 (4 GB VRAM) or equivalent
- Storage: 512 GB SSD

---

## 📁 Project Structure

```
traffic-sign-recognition/
├── data/
│   └── GTSRB/
│       ├── Train.csv
│       └── images/
├── notebooks/
│   └── TSR_Complete_Implementation.ipynb
├── models/
│   ├── mobilenet_model.h5
│   └── cnn_model.h5
├── results/
│   ├── confusion_matrices/
│   ├── clustering_visualizations/
│   └── performance_metrics/
├── README.md
└── requirements.txt
```

---

## 🔬 Methodology

### Architecture Overview

```
Input Images (GTSRB Dataset)
         ↓
   Preprocessing
   (Resize, Normalize, Augment)
         ↓
    ┌────────┴────────┐
    ↓                 ↓
Classification    Clustering
    ↓                 ↓
5 Models          5 Methods
    ↓                 ↓
Evaluation       Analysis
```

### 1. Preprocessing Pipeline

- **Resizing:** All images standardized to 64×64 pixels
- **Normalization:** Pixel values scaled to [0, 1]
- **Data Augmentation:**
  - Rotation (±8°)
  - Width/Height shift (±8%)
  - Zoom (±8%)
  - Brightness adjustment

### 2. Classification Models (5 Required)

| Model | Type | Feature Extraction | Accuracy |
|-------|------|-------------------|----------|
| **MobileNetV2** (Proposed) | Transfer Learning | Pre-trained CNN | **86.00%** |
| **SVM (Linear)** | Traditional ML | PCA (150 components) | **95.45%** |
| **K-NN** | Traditional ML | PCA (150 components) | 89.52% |
| **Decision Tree** | Traditional ML | PCA (150 components) | 55.20% |
| **Naive Bayes** | Traditional ML | PCA (150 components) | 43.36% |

**Note:** While SVM achieved highest accuracy (95.45%), MobileNetV2 was chosen as the proposed method due to:
- End-to-end learning capability
- No dependency on hand-crafted features
- Better adaptability and scalability
- Real-time deployment feasibility

### 3. Clustering Methods (5 Required)

| Method | Type | Silhouette Score | Davies-Bouldin Index |
|--------|------|------------------|---------------------|
| **K-Means** (Proposed) | Partitioning | 0.1885 | 1.6815 |
| **MiniBatchKMeans** | Partitioning | 0.1648 | 1.8471 |
| **Hierarchical (Agglomerative)** | Hierarchical | 0.1541 | 1.7529 |
| **DBSCAN** | Density-based | -0.1882 | 1.4079 |
| **Spectral** | Graph-based | 0.0288 | 1.4951 |

**Key Findings:**
- K-Means showed best cluster coherence
- Significant overlap between visually similar sign categories
- Clustering useful for identifying ambiguous samples
- Supervised learning necessary for accurate classification

### 4. Evaluation Metrics

**Classification Metrics:**
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1-Score (macro-averaged)
- Confusion Matrix

**Clustering Metrics:**
- Silhouette Score
- Davies-Bouldin Index
- Dunn Index
- WCSS (Within-Cluster Sum of Squares)

---

## 📈 Results

### Best Classification Performance

**MobileNetV2 (Proposed Method):**
- **Accuracy:** 86.00%
- **Precision:** 0.8596
- **Recall:** 0.8258
- **F1-Score:** 0.8262

**Comparison with Baseline CNN:**
- Custom CNN from scratch: 47.80% accuracy
- Transfer learning improvement: **+38.2%**

### Key Insights

1. **Transfer Learning Superiority:** Pre-trained MobileNetV2 significantly outperformed CNN trained from scratch
2. **SVM Trade-off:** While SVM achieved 95.45% accuracy, it requires fixed hand-crafted features
3. **Clustering Analysis:** Revealed challenges in distinguishing visually similar signs (speed limits, prohibitory signs)
4. **Real-world Applicability:** MobileNetV2 offers best balance for deployment in intelligent transportation systems

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

1. **Upload dataset to Google Drive:**
   ```
   Google Drive/GTSRB/archive.zip
   ```

2. **Open notebook in Colab and run cells sequentially:**
   - Mount Google Drive
   - Extract and preprocess dataset
   - Train classification models
   - Perform clustering analysis
   - Evaluate and visualize results

3. **Test with custom image:**
   - Run the demo section
   - Upload your traffic sign image
   - Get instant prediction

### Option 2: Local Environment

1. **Clone repository:**
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download GTSRB dataset:**
   - Place in `data/GTSRB/` directory

4. **Run the notebook:**
   ```bash
   jupyter notebook notebooks/TSR_Complete_Implementation.ipynb
   ```

### Quick Test with Pre-trained Model

```python
# Load the model
from tensorflow.keras.models import load_model
model = load_model('models/mobilenet_model.h5')

# Predict on new image
import cv2, numpy as np

img = cv2.imread('test_sign.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (64, 64)) / 255.0
img = np.expand_dims(img, axis=0)

prediction = np.argmax(model.predict(img))
print(f"Predicted ClassId: {prediction}")
```

---

## 🔮 Future Work

### Proposed Enhancements

1. **Dataset Enrichment:**
   - Integrate BelgiumTS and TT100K datasets
   - Enhance generalization across different regions

2. **Handling Class Imbalance:**
   - Implement advanced loss weighting
   - Use oversampling or synthetic sampling techniques

3. **Model Optimization:**
   - Apply pruning and quantization
   - Knowledge distillation for edge deployment

4. **Multi-modal Integration:**
   - Fuse with LiDAR, radar, or GPS data
   - Improve robustness in adverse conditions

5. **Real-Time Deployment:**
   - Test on embedded devices (Raspberry Pi, NVIDIA Jetson)
   - Optimize for low-latency inference

6. **Advanced Architectures:**
   - Experiment with EfficientNet, Vision Transformers
   - Implement attention mechanisms

---

## 📚 References

### Key Papers

1. G. Zhang et al., "A Traffic Sign Recognition System Based on Lightweight Network Learning," *Journal of Intelligent & Robotic Systems*, vol. 109, 2024.

2. M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *Proc. CVPR*, 2018.

3. J. Stallkamp et al., "The German Traffic Sign Recognition Benchmark: A Multi-class Classification Competition," *IEEE IJCNN*, 2011.

### Frameworks & Libraries

- TensorFlow/Keras: Deep learning framework
- Scikit-learn: Machine learning algorithms
- OpenCV: Image processing



.




