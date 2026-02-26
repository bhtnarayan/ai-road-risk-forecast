import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("data/nepal_road_accidents.csv")

# Preprocess
df_processed, label_encoders = preprocess_data(df)

# Target variable
y = df_processed["Severity"]
X = df_processed.drop("Severity", axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model + encoders
joblib.dump(model, "model/severity_model.pkl")
joblib.dump(label_encoders, "model/encoders.pkl")

print("Model trained and saved successfully!")