# =========================
# IMPORTS
# =========================
import numpy as np
import struct
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import zipfile

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# =========================
# EXTRACT ZIP
# =========================
zip_path = r"C:\Users\Rd\Downloads\archive (3).zip"
extract_path = r"C:\Users\Rd\Downloads\mnist_data"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)

print("Files:", os.listdir(extract_path))

# =========================
# LOAD FUNCTIONS
# =========================
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        return images.astype(np.float32) / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# =========================
# LOAD DATA (CORRECT PATH)
# =========================
train_images = load_mnist_images(os.path.join(extract_path, "train-images.idx3-ubyte"))
train_labels = load_mnist_labels(os.path.join(extract_path, "train-labels.idx1-ubyte"))

test_images = load_mnist_images(os.path.join(extract_path, "t10k-images.idx3-ubyte"))
test_labels = load_mnist_labels(os.path.join(extract_path, "t10k-labels.idx1-ubyte"))

print("Train shape:", train_images.shape)
print("Test shape:", test_images.shape)

# =========================
# VISUALIZATION
# =========================
def plot_mnist_grid(images, labels, grid_size=5):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(str(labels[i]))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

plot_mnist_grid(train_images, train_labels)

# =========================
# PREPROCESSING
# =========================
X_train = train_images.reshape(train_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

y_train = train_labels
y_test = test_labels

# =========================
# MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# =========================
# CROSS VALIDATION
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

train_report = {}
test_report = {}

# =========================
# TRAIN + EVALUATE
# =========================
for name, model in models.items():
    print(f"\nTraining {name}...")

    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scorers)

    train_report[name] = {
        'accuracy': np.mean(cv_results['test_accuracy']),
        'precision': np.mean(cv_results['test_precision']),
        'recall': np.mean(cv_results['test_recall']),
        'f1': np.mean(cv_results['test_f1'])
    }

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_report[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }

# =========================
# RESULTS
# =========================
train_df = pd.DataFrame(train_report).T.round(4)
test_df = pd.DataFrame(test_report).T.round(4)

print("\n=== TRAINING (CV) ===")
print(train_df)

print("\n=== TEST RESULTS ===")
print(test_df)

# =========================
# CONFUSION MATRIX
# =========================
best_model_name = test_df['accuracy'].idxmax()
print("\nBest Model:", best_model_name)

best_model = models[best_model_name]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# GRAPH
# =========================
train_df.plot(kind='bar', figsize=(10,6))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()