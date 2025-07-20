from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("student_performance_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# Define section-wise input models

class StudentDetails(BaseModel):
    school: str
    sex: str
    age: int
    address: str

class FamilyBackground(BaseModel):
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    famsup: str

class AcademicStatus(BaseModel):
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    paid: str
    activities: str
    internet: str
    nursery: str
    higher: str
    absences: int
    G1: int
    G2: int

class HealthStatus(BaseModel):
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int

class StudentInput(BaseModel):
    student: StudentDetails
    family: FamilyBackground
    academic: AcademicStatus
    health: HealthStatus

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Performance Prediction API!"}

@app.post("/predict")
def predict(data: StudentInput):
    try:
        # Flatten nested input dicts
        input_data = {
            **data.student.dict(),
            **data.family.dict(),
            **data.academic.dict(),
            **data.health.dict()
        }

        # Apply label encoding where needed
        for col in label_encoders:
            le = label_encoders[col]
            if input_data[col] not in le.classes_:
                return {
                    "error": f"Unknown value '{input_data[col]}' for field '{col}'. Allowed: {list(le.classes_)}"
                }
            input_data[col] = le.transform([input_data[col]])[0]

        # Create DataFrame in model's expected format
        df_input = pd.DataFrame([input_data])[model.feature_names_in_]

        # Predict final grade
        prediction = model.predict(df_input)[0]
        return {"predicted_final_Marks": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
