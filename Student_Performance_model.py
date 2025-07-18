import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('D:\Sparsh\ML_Projects\Student_Performance_Prediction\Dataset\student_performance_dataset.csv')

# List of categorical columns
categorical_cols = ['parent_education', 'extra_activities', 'course_done']

# Dictionary to store encoders
label_encoders = {}

# Apply label encoding
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df.drop(columns=['final_grade'])
y = df['final_grade']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save model and label encoder
joblib.dump(model, 'student_performance_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
print("Model and encoder saved.")
