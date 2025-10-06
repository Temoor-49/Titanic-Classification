import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(df, is_train=True, save_encoders=False):
    """
    Preprocess Titanic data with consistent feature engineering.
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data (to fit encoders)
        save_encoders: Whether to save encoders (only for training)
    
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # ===== Handle Missing Values =====
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # ===== Feature Engineering =====
    
    # 1. Extract Title from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(
        ['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Sir', 'Don', 'Jonkheer'], 'Rare'
    ).replace(['Lady', 'Countess', 'Dona'], 'Rare')
    
    # 2. Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 3. Age Bins
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 80], labels=False)
    
    # 4. Fare Bins
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=False, duplicates='drop')
    
    # Fill any NaNs from binning
    df['AgeBin'] = df['AgeBin'].fillna(df['AgeBin'].mode()[0])
    df['FareBin'] = df['FareBin'].fillna(df['FareBin'].mode()[0])
    
    # ===== Encode Categorical Variables =====
    
    if is_train:
        # Fit encoders on training data
        sex_encoder = LabelEncoder()
        embarked_encoder = LabelEncoder()
        title_encoder = LabelEncoder()
        
        df['Sex'] = sex_encoder.fit_transform(df['Sex'])
        df['Embarked'] = embarked_encoder.fit_transform(df['Embarked'])
        df['Title'] = title_encoder.fit_transform(df['Title'])
        
        # Save encoders for later use
        if save_encoders:
            os.makedirs('models', exist_ok=True)
            joblib.dump(sex_encoder, 'models/sex_encoder.joblib')
            joblib.dump(embarked_encoder, 'models/embarked_encoder.joblib')
            joblib.dump(title_encoder, 'models/title_encoder.joblib')
    else:
        # Load and apply encoders
        sex_encoder = joblib.load('models/sex_encoder.joblib')
        embarked_encoder = joblib.load('models/embarked_encoder.joblib')
        title_encoder = joblib.load('models/title_encoder.joblib')
        
        # Handle unseen titles
        df['Title'] = df['Title'].apply(
            lambda x: x if x in title_encoder.classes_ else 'Rare'
        )
        
        df['Sex'] = sex_encoder.transform(df['Sex'])
        df['Embarked'] = embarked_encoder.transform(df['Embarked'])
        df['Title'] = title_encoder.transform(df['Title'])
    
    # ===== Drop Unused Columns =====
    columns_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df