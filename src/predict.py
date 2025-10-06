import pandas as pd
import joblib
from preprocess import preprocess_data

def predict_new_passengers(new_data: pd.DataFrame, model_name='logistic_regression'):
    """
    Predict survival for new passenger(s).
    
    Args:
        new_data (pd.DataFrame): passenger data in same format as Titanic dataset
        model_name (str): name of the model file (without .joblib)
    
    Returns:
        numpy.array: predictions (0 = did not survive, 1 = survived)
    """
    print(f"Preprocessing {len(new_data)} passenger(s)...")
    clean_data = preprocess_data(new_data, is_train=False, save_encoders=False)
    
    print(f"Loading model: {model_name}...")
    model = joblib.load(f"models/{model_name}.joblib")
    
    print(f"Making predictions...")
    predictions = model.predict(clean_data)
    
    return predictions

if __name__ == "__main__":
    # Example: multiple passengers
    example = pd.DataFrame([
        {
            "PassengerId": 1001,
            "Pclass": 1,
            "Name": "Brown, Mrs. Alice",
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
            "Name": "Smith, Mr. Bob",
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
            "Name": "Johnson, Mr. Charlie",
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
    
    # Make predictions
    preds = predict_new_passengers(example, model_name='logistic_regression')
    
    # Display results nicely
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    
    for i, row in example.iterrows():
        status = 'Survived ✅' if preds[i] == 1 else 'Did not survive ❌'
        print(f"{row['Name']:30} | {status}")
    
    # Save results to CSV
    output = example.copy()
    output['Prediction'] = preds
    output['Prediction_Label'] = output['Prediction'].map({0: 'Did not survive', 1: 'Survived'})
    output.to_csv('predictions.csv', index=False)
    
    print(f"\n✅ Predictions saved to predictions.csv")