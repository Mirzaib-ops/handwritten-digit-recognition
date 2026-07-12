"""
Task 2: Handwritten Digit Recognition
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load & Visualize
digits = load_digits()
X = digits.data
y = digits.target

print(f"Total Images: {len(X)}")
print(f"Image Shape : {digits.images[0].shape}")

fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, img, label in zip(axes, digits.images, y):
    ax.set_axis_off()
    ax.imshow(img, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Label: {label}")
plt.suptitle("Sample Digits")
plt.savefig("task2_sample_digits.png", dpi=150, bbox_inches="tight")
plt.show()

# Step 2: Split & Scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data successfully split and scaled!")

# Step 3: Train SVM & Random Forest
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Evaluate
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"SVM Accuracy: {svm_acc * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")

# Step 5: Confusion Matrix & Misclassified Samples
cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("task2_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

misclassified_idx = np.where(svm_pred != y_test)[0]
if len(misclassified_idx) > 0:
    n_show = min(5, len(misclassified_idx))
    fig, axes = plt.subplots(1, n_show, figsize=(10, 3))
    if n_show == 1:
        axes = [axes]
    for ax, idx in zip(axes, misclassified_idx[:n_show]):
        ax.imshow(X_test[idx].reshape(8, 8), cmap="gray_r")
        ax.set_title(f"True: {y_test[idx]}\nPred: {svm_pred[idx]}")
        ax.set_axis_off()
    plt.suptitle("Misclassified Digits (SVM)")
    plt.savefig("task2_misclassified.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nSummary")
print(f"SVM (RBF): {svm_acc*100:.2f}% accuracy | {int(round((1-svm_acc)*len(y_test)))} errors out of {len(y_test)}")
print(f"Random Forest: {rf_acc*100:.2f}% accuracy | {int(round((1-rf_acc)*len(y_test)))} errors out of {len(y_test)}")
