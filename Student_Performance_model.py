import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('D:\Sparsh\ML_Projects\Student_Performance_Prediction\Dataset\student_performance_portuguese_dataset.csv') 

# Target
y = df['G3']
X = df.drop(columns=['G3'])

# Detect and encode categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save model and encoders
joblib.dump(model, 'student_performance_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
print("Model and encoders saved.")
