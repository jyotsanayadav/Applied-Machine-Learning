import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
#Load Dataset
df = pd.read_csv("C:\\Users\\Rd\\Downloads\\archive (2).zip")
print("Initial Shape:", df.shape)
print(df.info())
# Data Cleaning
# Remove customerID
df.drop("customerID", axis=1, inplace=True)
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
# Check Class Distribution
print("\nClass Distribution (%):")
print(df["Churn"].value_counts(normalize=True) * 100)
# Outlier Detection (Boxplots)
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()
# Feature Engineering
# Label Encoding for Binary Columns
binary_cols = ["gender", "Partner", "Dependents",
               "PhoneService", "PaperlessBilling", "Churn"]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
# One-Hot Encoding for remaining categorical columns
df = pd.get_dummies(df, drop_first=True)
# Normalization using Min-Max Scaling
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
# Train-Test Split (75% - 25%)
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=True
)
print("\nTraining Size:", X_train.shape)
print("Testing Size:", X_test.shape)

#  Mutual Information (Feature Importance)
mi_scores = mutual_info_classif(X, y)
mi = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(mi.head(10))

plt.figure(figsize=(8,6))
mi.head(10).plot(kind='barh')
plt.title("Top 10 Important Features (Mutual Information)")
plt.show()

# Model Training

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Support Vector Machine
svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model Evaluation

print("\n===== Model Accuracies =====")
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("SVC:", accuracy_score(y_test, y_pred_svc))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))

print("\n===== Classification Report (Random Forest) =====")
print(classification_report(y_test, y_pred_rf))

# Final Conclusion
print("\nConclusion:")
print("Random Forest generally provides the best performance due to ensemble learning.")
print("Contract type, tenure, and monthly charges are strong churn predictors.")
