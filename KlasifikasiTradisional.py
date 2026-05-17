import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    learning_curve,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix
from skimage.feature import graycoprops

DATASET_PATH = "dataset"

IMAGE_SIZE = (128, 128)


def load_dataset(dataset_path):

    images = []
    labels = []

    classes = os.listdir(dataset_path)

    for label in classes:

        class_path = os.path.join(dataset_path, label)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):

            if file.endswith((".jpg", ".png", ".jpeg")):

                image_path = os.path.join(class_path, file)

                image = cv2.imread(image_path)

                if image is None:
                    continue

                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(label)

    return images, labels

def extract_hog(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        visualize=False
    )

    return features


def extract_color_histogram(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0,1,2],
        None,
        [8,8,8],
        [0,180,0,256,0,256]
    )

    hist = cv2.normalize(hist, hist).flatten()

    return hist

def extract_hu_moments(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    moments = cv2.moments(gray)

    hu = cv2.HuMoments(moments).flatten()

    return hu

def extract_glcm(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]

    return np.array([
        contrast,
        correlation,
        energy,
        homogeneity
    ])

def extract_features(images):

    feature_list = []

    for image in images:

        hog_feat = extract_hog(image)

        color_feat = extract_color_histogram(image)

        hu_feat = extract_hu_moments(image)

        glcm_feat = extract_glcm(image)

        combined = np.hstack([
            hog_feat,
            color_feat,
            hu_feat,
            glcm_feat
        ])

        feature_list.append(combined)

    return np.array(feature_list)

def evaluate_model(model, X_test, y_test):

    start = time.time()

    y_pred = model.predict(X_test)

    inference_time = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "inference_time": inference_time,
        "predictions": y_pred
    }

def plot_confusion_matrix(cm, classes, title):

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

def visualize_decision_boundary(
    X,
    y,
    model,
    title
):

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    model.fit(X_pca, y)

    x_min, x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
    y_min, y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    Z = model.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))

    plt.contourf(xx, yy, Z, alpha=0.3)

    scatter = plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=y
    )

    plt.title(title)

    plt.show()

def plot_learning_curve(
    estimator,
    X,
    y,
    title
):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8,6))

    plt.plot(train_sizes, train_mean, label='Training Score')

    plt.plot(train_sizes, test_mean, label='Validation Score')

    plt.title(title)

    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.show()

if __name__ == "__main__":

    print("="*50)
    print("LOADING DATASET")
    print("="*50)

    images, labels = load_dataset(DATASET_PATH)

    print("Jumlah data :", len(images))

    encoder = LabelEncoder()

    y = encoder.fit_transform(labels)


    print("="*50)
    print("EXTRACTING FEATURES")
    print("="*50)

    X = extract_features(images)

    print("Feature shape :", X.shape)

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("="*50)
    print("KNN EXPERIMENT")
    print("="*50)

    k_values = [1,3,5,7,9,11]

    distance_metrics = [
        'euclidean',
        'manhattan',
        'minkowski'
    ]

    best_knn = None
    best_knn_acc = 0

    for k in k_values:

        for metric in distance_metrics:

            start = time.time()

            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric=metric
            )

            knn.fit(X_train, y_train)

            training_time = time.time() - start

            result = evaluate_model(
                knn,
                X_test,
                y_test
            )

            print(f"""
K = {k}
Metric = {metric}
Accuracy = {result['accuracy']:.4f}
Precision = {result['precision']:.4f}
Recall = {result['recall']:.4f}
F1-score = {result['f1']:.4f}
Training Time = {training_time:.4f}
Inference Time = {result['inference_time']:.4f}
            """)

            if result['accuracy'] > best_knn_acc:

                best_knn_acc = result['accuracy']

                best_knn = knn

                best_knn_cm = result['confusion_matrix']
    print("="*50)
    print("SVM EXPERIMENT")
    print("="*50)

    kernels = ['linear', 'poly', 'rbf']

    C_values = [0.1, 1, 10, 100]

    gamma_values = [0.001, 0.01, 0.1, 1]

    best_svm = None
    best_svm_acc = 0

    for kernel in kernels:

        for C in C_values:

            for gamma in gamma_values:

                if kernel != 'rbf':
                    gamma = 'scale'

                start = time.time()

                svm = SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    probability=True
                )

                svm.fit(X_train, y_train)

                training_time = time.time() - start

                result = evaluate_model(
                    svm,
                    X_test,
                    y_test
                )

                print(f"""
Kernel = {kernel}
C = {C}
Gamma = {gamma}
Accuracy = {result['accuracy']:.4f}
Precision = {result['precision']:.4f}
Recall = {result['recall']:.4f}
F1-score = {result['f1']:.4f}
Training Time = {training_time:.4f}
Inference Time = {result['inference_time']:.4f}
                """)

                if result['accuracy'] > best_svm_acc:

                    best_svm_acc = result['accuracy']

                    best_svm = svm

                    best_svm_cm = result['confusion_matrix']

    class_names = encoder.classes_

    plot_confusion_matrix(
        best_knn_cm,
        class_names,
        "Best KNN Confusion Matrix"
    )

    plot_confusion_matrix(
        best_svm_cm,
        class_names,
        "Best SVM Confusion Matrix"
    )
    print("="*50)
    print("CROSS VALIDATION")
    print("="*50)

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    knn_cv = cross_val_score(
        best_knn,
        X,
        y,
        cv=skf
    )

    svm_cv = cross_val_score(
        best_svm,
        X,
        y,
        cv=skf
    )

    print("KNN CV Accuracy :", knn_cv.mean())

    print("SVM CV Accuracy :", svm_cv.mean())

    plot_learning_curve(
        best_knn,
        X,
        y,
        "KNN Learning Curve"
    )

    plot_learning_curve(
        best_svm,
        X,
        y,
        "SVM Learning Curve"
    )
    print("="*50)
    print("GRID SEARCH KNN")
    print("="*50)

    knn_params = {
        'n_neighbors': [1,3,5,7,9,11],
        'metric': [
            'euclidean',
            'manhattan',
            'minkowski'
        ]
    }

    grid_knn = GridSearchCV(
        KNeighborsClassifier(),
        knn_params,
        cv=5
    )

    grid_knn.fit(X_train, y_train)

    print("Best KNN Params :")
    print(grid_knn.best_params_)

    print("Best KNN Score :")
    print(grid_knn.best_score_)
    print("="*50)
    print("GRID SEARCH SVM")
    print("="*50)

    svm_params = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': [0.1,1,10,100],
        'gamma': [0.001,0.01,0.1,1]
    }

    grid_svm = GridSearchCV(
        SVC(),
        svm_params,
        cv=5
    )

    grid_svm.fit(X_train, y_train)

    print("Best SVM Params :")
    print(grid_svm.best_params_)

    print("Best SVM Score :")
    print(grid_svm.best_score_)

    visualize_decision_boundary(
        X_train,
        y_train,
        KNeighborsClassifier(n_neighbors=5),
        "KNN Decision Boundary"
    )

    visualize_decision_boundary(
        X_train,
        y_train,
        SVC(kernel='rbf'),
        "SVM Decision Boundary"
    )