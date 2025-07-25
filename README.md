# 🤖 KNN & SVM From Scratch (Single Python File | Core Python Only)

This repository contains **simple, from-scratch implementations** of two foundational machine learning algorithms:

- 📌 **K-Nearest Neighbors (KNN)**
- 📌 **Support Vector Machine (SVM)** (Hard Margin)

Both are implemented using **basic Python constructs only**, without any third-party libraries like `scikit-learn`, `numpy`, or `pandas`.

---

## 📁 File Structure

```

KNN_SVM_From_Scratch.ipynb      # Combined implementation of KNN and SVM
README.md       # Project documentation

````

---

## 🔍 Overview

| Algorithm | Type             | Usage                | Key Concept                    |
|-----------|------------------|----------------------|--------------------------------|
| KNN       | Instance-based   | Classification       | Distance from nearest neighbors |
| SVM       | Model-based      | Classification       | Maximize margin using hyperplane |

---

## 📌 What is KNN?

**K-Nearest Neighbors** is a simple algorithm that:

- Stores the training data
- Classifies new points based on the **majority label of the k nearest points**

### ✅ Key Points:
- Lazy learning (no training phase)
- Distance-based voting
- Simple to implement and interpret

### ⚠️ Limitations:
- Slow at prediction time (especially on large data)
- Sensitive to irrelevant features and noise
- Requires scaling if features differ in units

---

## 📌 What is SVM?

**Support Vector Machine** is a supervised learning algorithm that:

- Learns a **hyperplane (decision boundary)** that best separates two classes
- Maximizes the margin between classes
- Uses **only support vectors** to define this boundary

### ✅ Key Points:
- Strong generalization ability
- Effective in high-dimensional spaces
- Can be extended to non-linear problems with **kernels** (not included here)

### ⚠️ Limitations:
- Training can be slow (especially with brute-force search like here)
- Sensitive to outliers
- Needs good choice of parameters (not tunable in basic version)

---

## 🧠 How They Work (Internally)

### 🔸 KNN:
1. Compute distances between test point and all training points
2. Sort and pick `k` closest
3. Perform majority vote to predict class

### 🔸 SVM:
1. Brute-force search for weights `w` and bias `b`
2. Ensure all points satisfy:  
 \[
y_i (w \cdot x_i + b) \geq 1
\]

3. Choose the line that **maximizes the margin** = `1 / ||w||`

---

## 🔄 KNN vs. SVM

| Feature               | KNN                          | SVM (Hard Margin)               |
|----------------------|-------------------------------|----------------------------------|
| Training Time        | None                          | High (search-based optimization) |
| Prediction Time      | Slow (computes distances)     | Fast (dot product with w, b)     |
| Memory Usage         | High                          | Low (only support vectors)       |
| Sensitivity to Noise | High                          | Medium                           |
| Handles Non-linearity| ❌ Not in basic form          | ✅ With kernels (not implemented here) |
| Use Case             | Small, clean datasets         | Complex, high-dimensional data   |

---

## 🧪 Example Usage (from `knn_svm.py`)

### 🔹 KNN Example

```
knn_dataset = [
    ([1, 2], 'A'),
    ([2, 3], 'A'),
    ([3, 3], 'B'),
    ([6, 5], 'B')
]
test_point = [2.5, 2.7]
k = 3
print(knn_predict(knn_dataset, test_point, k))
```

### 🔹 SVM Example

```
svm_data = {
    1: [[2, 3], [3, 4], [4, 5]],
   -1: [[6, 1], [7, 3], [8, -1]]
}
w, b = train_svm(svm_data)
test_point = [5, 3]
print(predict_svm(w, b, test_point))
```

---

## 🧩 What You'll Learn

* How classification works from scratch
* The role of distance in KNN
* How SVM separates data using geometry (dot products, margins)
* The difference between **lazy vs. eager learning**
* Why model design matters (KNN stores, SVM learns)

---

## 🎯 Goals of This Project

* ✅ Learn the **fundamentals** of KNN and SVM
* ✅ Understand classification logic without libraries
* ✅ Build ML confidence from the ground up
* ✅ Serve as a base for adding more algorithms (Logistic, Naive Bayes, etc.)

---


## 🙌 Contributions

Pull requests are welcome!
Whether it’s adding features (e.g., soft margin, kernels, visualizations) or improving clarity — feel free to contribute!

