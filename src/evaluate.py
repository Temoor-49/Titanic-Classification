import pandas as pd
import joblib
from preprocess import preprocess_data

def evaluate_model(model_name='logistic_regression'):
    """
    Generate predictions on test set and create submission file.
    
    Args:
        model_name: Name of the model file (without .joblib)
    """
    print(f"Loading test data...")
    test_df = pd.read_csv("data/test.csv")
    passenger_ids = test_df['PassengerId'].copy()
    
    print(f"Preprocessing test data...")
    test_clean = preprocess_data(test_df, is_train=False, save_encoders=False)
    
    print(f"Loading model: {model_name}...")
    model = joblib.load(f"models/{model_name}.joblib")
    
    print(f"Making predictions...")
    predictions = model.predict(test_clean)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print(f"✅ Submission file saved as submission.csv")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted survivors: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
    print(f"Predicted non-survivors: {len(predictions)-sum(predictions)} ({(len(predictions)-sum(predictions))/len(predictions)*100:.1f}%)")

if __name__ == "__main__":
    # You can change the model name here
    evaluate_model(model_name='logistic_regression')

