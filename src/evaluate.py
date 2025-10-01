# src/evaluate.py
import pandas as pd
import joblib
from preprocess import preprocess_data

def evaluate_model():
    # Load test set
    test_df = pd.read_csv("data/test.csv")
    passenger_ids = test_df["PassengerId"]  # save IDs for submission

    # Preprocess test set
    test_clean = preprocess_data(test_df, is_train=False)

    # Load trained model
    model = joblib.load("models/logistic_regression.joblib")

    # Predict
    predictions = model.predict(test_clean)

    # Create submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })

    # Save submission
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission file saved as ../submission.csv")

if __name__ == "__main__":
    evaluate_model()
