from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("D:\Sparsh\ML_Projects\Student_Performance_Prediction\student_performance_model.joblib")
label_encoders = joblib.load("D:\Sparsh\ML_Projects\Student_Performance_Prediction\label_encoders.joblib")

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
    class_test_performance: float

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(data: StudentInput):
    try:
        input_data = data.dict()

        # Apply label encoders to the necessary fields
        for col in ['parent_education', 'extra_activities', 'course_done']:
            encoder = label_encoders[col]
            if input_data[col] not in encoder.classes_:
                return {"error": f"Unknown category '{input_data[col]}' for field '{col}'."}
            input_data[col] = encoder.transform([input_data[col]])[0]

        column_order = [
            'hours_studied',
            'attendance',
            'parent_education',
            'extra_activities',
            'course_done',
            'course_score',
            'put',
            'st1',
            'st2',
            'class_test_performance'
        ]

        df_input = pd.DataFrame([[input_data[col] for col in column_order]], columns=column_order)

        prediction = model.predict(df_input)[0]
        return {"predicted_final_grade": round(prediction, 2)}
    
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
