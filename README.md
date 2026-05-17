# Komparasi Klasifikasi KNN vs SVM untuk Pengenalan Objek Citra

Project computer vision untuk membandingkan performa algoritma **K-Nearest Neighbors (KNN)** dan **Support Vector Machine (SVM)** dalam klasifikasi citra objek menggunakan berbagai metode ekstraksi fitur.

---

## Fitur yang Digunakan

- HOG (Histogram of Oriented Gradients)
- Color Histogram (HSV)
- Hu Moments
- GLCM Texture

---

## Algoritma Klasifikasi

### KNN
- k = 1, 3, 5, 7, 9, 11
- Euclidean Distance
- Manhattan Distance
- Minkowski Distance

### SVM
- Linear Kernel
- Polynomial Kernel
- RBF Kernel

---

## Evaluasi

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Cross Validation
- Learning Curve
- Decision Boundary Visualization

---

## Struktur Dataset

```text
dataset/
├── buku/
├── botol/
├── mug/
├── remote/
└── mainan/
```

---

## Install Library

```bash
pip install opencv-python scikit-learn matplotlib seaborn numpy scikit-image
```

---

## Menjalankan Program

```bash
python main.py
```

---

## Tools & Library

- Python
- OpenCV
- Scikit-Learn
- NumPy
- Matplotlib
- Seaborn
- Scikit-Image

---

## Author

Nama: YOUR_NAME  
Mata Kuliah: Computer Vision / Pengolahan Citra
