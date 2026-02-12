
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("=== Logistic Regression Results ===")
print(f"{accuracy_score(y_test, y_pred_lr):.4f}")
print(f"{precision_score(y_test, y_pred_lr):.4f}")
print(f"{recall_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=['malignant', 'benign']))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"{accuracy_score(y_test, y_pred_rf):.4f}")
print(f"{precision_score(y_test, y_pred_rf):.4f}")
print(f"{recall_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf, target_names=['malignant', 'benign']))


np.random.seed(42)

image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
y, x = np.ogrid[0:rows, 0:cols]
radius = 20
mask = (x - center_col)**2 + (y - center_row)**2 <= radius**2
image[mask] = 200

image_norm     = image.astype(np.float32) / 255.0
image_denoised = gaussian_filter(image_norm, sigma=1.0)

image_enhanced = np.clip(
    (image_denoised - image_denoised.min()) / 
    (image_denoised.max() - image_denoised.min() + 1e-8), 
    0, 1
)

sobel_x = sobel(image_enhanced, axis=0)
sobel_y = sobel(image_enhanced, axis=1)
image_edges = np.hypot(sobel_x, sobel_y)  

threshold = 0.4
image_binary = (image_edges > threshold).astype(np.float32)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Synthetic Medical Image Preprocessing Stages", fontsize=14)

axes[0,0].imshow(image, cmap='gray')
axes[0,0].set_title("Original")
axes[0,0].axis('off')

axes[0,1].imshow(image_denoised, cmap='gray')
axes[0,1].set_title("Denoised (Gaussian σ=1)")
axes[0,1].axis('off')

axes[0,2].imshow(image_enhanced, cmap='gray')
axes[0,2].set_title("Contrast Enhanced")
axes[0,2].axis('off')

axes[1,0].imshow(image_edges, cmap='gray')
axes[1,0].set_title("Edges (Sobel)")
axes[1,0].axis('off')

axes[1,1].imshow(image_binary, cmap='gray')
axes[1,1].set_title(f"Binary (threshold={threshold})")
axes[1,1].axis('off')

axes[1,2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

