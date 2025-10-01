# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, is_train=True):
    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fill missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fill missing Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Drop Cabin (too many missing)
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])

    # Convert Sex to numeric
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Encode Embarked
    encoder = LabelEncoder()
    df['Embarked'] = encoder.fit_transform(df['Embarked'])

    # Drop Ticket, Name, PassengerId
    df = df.drop(columns=['Ticket', 'Name', 'PassengerId'], errors='ignore')

    return df
