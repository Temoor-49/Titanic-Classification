import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from preprocess import preprocess_data

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load training data
print("Loading training data...")
data = pd.read_csv("data/train.csv")

# Preprocess data (save encoders for later use)
print("Preprocessing data...")
data = preprocess_data(data, is_train=True, save_encoders=True)

# Features and labels
X = data.drop(columns=['Survived'])
y = data['Survived']

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
print(f"Features: {list(X.columns)}\n")

# Models to compare
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=5),
}

best_model = None
best_acc = 0
best_name = ""

# Train & evaluate each model
for name, model in models.items():
    print(f"{'='*50}")
    print(f"Training {name.replace('_', ' ').title()}...")
    print(f"{'='*50}")
    
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_val, preds, target_names=['Did not survive', 'Survived']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_val, preds))
    
    # Save each model
    joblib.dump(model, f"models/{name}.joblib")
    print(f"✅ Model saved: models/{name}.joblib\n")
    
    # Track best model
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\n{'='*50}")
print(f"🏆 Best Model: {best_name.replace('_', ' ').title()}")
print(f"🏆 Best Accuracy: {best_acc:.4f}")
print(f"{'='*50}")