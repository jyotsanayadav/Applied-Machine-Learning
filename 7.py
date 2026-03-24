# Exp. 7: Iris Flower Classification
# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# 2. Load Dataset
import zipfile

with zipfile.ZipFile("C:\\Users\\Rd\\Downloads\\archive (5).zip") as z:
    print(z.namelist())  # see files
    
    with z.open("Iris.csv") as f:
        df = pd.read_csv(f)
#iris = load_iris()
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print("Feature Names:", df.columns[:-1].tolist())
print("Target Names:", df.iloc[:, -1].unique())
# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 4. Initialize Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}
# 5. Train & Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = acc
    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
# 6. Compare Results
print("\nModel Comparison:")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}")
# 7. Visualization (Accuracy Comparison)
plt.bar(results.keys(), results.values())
plt.xticks(rotation=30)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()