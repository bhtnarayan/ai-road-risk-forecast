import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    df = df.copy()

    # Drop unnecessary columns
    df = df.drop(["Accident_ID", "Accident_Date"], axis=1, errors="ignore")

    # Fill missing values
    df.fillna("Unknown", inplace=True)

    # Encode categorical columns
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders