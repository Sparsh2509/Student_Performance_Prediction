from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("D:\Sparsh\ML_Projects\Student_Performance_Prediction\student_performance_model.joblib")
label_encoders = joblib.load("D:\Sparsh\ML_Projects\Student_Performance_Prediction\label_encoder.joblib")

# Define request schema
class StudentInput(BaseModel):
    hours_studied: float
    attendance: float
    parent_education: str
    extra_activities: str
    course_done: str
    course_score: float
    put: float
    st1: float
    st2: float
    ct: float

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict_final_grade(data: StudentInput):
    try:
        input_data = data.dict()
        df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in ['parent_education', 'extra_activities', 'course_done']:
            if col in label_encoders:
                le = label_encoders[col]
                if df[col].iloc[0] not in le.classes_:
                    raise HTTPException(status_code=400, detail=f"Invalid value for {col}: {df[col].iloc[0]}")
                df[col] = le.transform(df[col])

        # Predict
        prediction = model.predict(df)[0]
        return {"predicted_final_grade": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
