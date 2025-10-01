# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load training data
data = pd.read_csv("data/train.csv")

# ===== Feature Engineering =====
# Fill missing values first
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Extract Title from Name
data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
data["Title"] = data["Title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs")
data["Title"] = data["Title"].replace(
    ["Dr", "Rev", "Col", "Major", "Capt", "Sir", "Don", "Jonkheer"], "Rare"
).replace(["Lady", "Countess", "Dona"], "Rare")
data["Title"] = LabelEncoder().fit_transform(data["Title"])

# Family size
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

# Age bins
data["AgeBin"] = pd.cut(data["Age"], bins=[0, 12, 18, 35, 50, 80], labels=False)

# Fare bins
data["FareBin"] = pd.qcut(data["Fare"], 4, labels=False)

# Handle any NaNs left
data = data.fillna({
    "AgeBin": data["AgeBin"].mode()[0],
    "FareBin": data["FareBin"].mode()[0],
    "Embarked": data["Embarked"].mode()[0]
})

# Drop unused columns
data = data.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])

# Encode categorical variables
for col in ["Sex", "Embarked"]:
    data[col] = LabelEncoder().fit_transform(data[col])

# Features and labels
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
}

best_model = None
best_acc = 0
best_name = ""

# Train & evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_val, preds))

    # Save best model
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# Save the best model
joblib.dump(best_model, f"models/{best_name.replace(' ', '_').lower()}.joblib")
print(f"\n✅ Best model saved: {best_name} with accuracy {best_acc:.4f}")
