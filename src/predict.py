# src/predict.py
import os
import pandas as pd
import joblib
from preprocess import preprocess_data

def predict_new_passengers(new_data: pd.DataFrame):
    """
    Predict survival for new passenger(s).
    Args:
        new_data (pd.DataFrame): passenger data in same format as Titanic dataset
    Returns:
        list: predictions (0 = did not survive, 1 = survived)
    """
    # Preprocess
    clean_data = preprocess_data(new_data, is_train=False)

    # Load model
    model = joblib.load("models/logistic_regression.joblib")

    # Predict
    predictions = model.predict(clean_data)
    return predictions

if __name__ == "__main__":
    # Example: multiple passengers
    example = pd.DataFrame([
        {
            "PassengerId": 1001,
            "Pclass": 1,
            "Name": "Alice Brown",
            "Sex": "female",
            "Age": 25,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "PC 17599",
            "Fare": 71.28,
            "Cabin": "C85",
            "Embarked": "C"
        },
        {
            "PassengerId": 1002,
            "Pclass": 3,
            "Name": "Bob Smith",
            "Sex": "male",
            "Age": 30,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": None,
            "Embarked": "S"
        },
        {
            "PassengerId": 1003,
            "Pclass": 2,
            "Name": "Charlie Johnson",
            "Sex": "male",
            "Age": 40,
            "SibSp": 0,
            "Parch": 1,
            "Ticket": "STON/O2. 3101282",
            "Fare": 13.00,
            "Cabin": None,
            "Embarked": "Q"
        }
    ])

    preds = predict_new_passengers(example)

    # Print nicely
    for i, row in example.iterrows():
        print(f"Passenger {row['Name']} (ID {row['PassengerId']}): "
              f"{'Survived' if preds[i] == 1 else 'Did not survive'}")
        
        # At the end of src/predict.py
    # Save results to CSV
    output = example.copy()
    output["Prediction"] = preds
    output["Prediction"] = output["Prediction"].map({0: "Did not survive", 1: "Survived"})
    output.to_csv("../predictions.csv", index=False)

    print("\nPredictions saved to predictions.csv ✅")

