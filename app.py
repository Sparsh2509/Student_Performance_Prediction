from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("student_performance_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# Sectioned input models
class StudentDetails(BaseModel):
    school: str = Field(...)
    sex: str = Field(...)
    age: int = Field(..., ge=15, le=22)
    address: str = Field(...)

class FamilyBackground(BaseModel):
    famsize: str = Field(...)
    Pstatus: str = Field(...)
    Medu: int = Field(..., ge=0, le=4)
    Fedu: int = Field(..., ge=0, le=4)
    Mjob: str = Field(...)
    Fjob: str = Field(...)
    reason: str = Field(...)
    guardian: str = Field(...)
    famsup: str = Field(...)

class AcademicStatus(BaseModel):
    traveltime: int = Field(..., ge=1, le=4)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0, le=4)
    schoolsup: str = Field(...)
    paid: str = Field(...)
    activities: str = Field(...)
    internet: str = Field(...)
    nursery: str = Field(...)
    higher: str = Field(...)
    absences: int = Field(..., ge=0, le=93)
    G1: int = Field(..., ge=0, le=20)
    G2: int = Field(..., ge=0, le=20)

class HealthStatus(BaseModel):
    romantic: str = Field(...)
    famrel: int = Field(..., ge=1, le=5)
    freetime: int = Field(..., ge=1, le=5)
    goout: int = Field(..., ge=1, le=5)
    Dalc: int = Field(..., ge=1, le=5)
    Walc: int = Field(..., ge=1, le=5)
    health: int = Field(..., ge=1, le=5)

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

        # Convert all string inputs to lowercase
        input_data = {k: v.lower() if isinstance(v, str) else v for k, v in input_data.items()}

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

        # Generate remark
        def get_remark(score):
            if score >= 18:
                return "A+ (Excellent)"
            elif score >= 16:
                return "A (Very Good)"
            elif score >= 14:
                return "B+ (Good)"
            elif score >= 12:
                return "B (Average)"
            elif score >= 10:
                return "C (Below Avg)"
            else:
                return "D (Poor)"

        remark = get_remark(prediction)

        return {
            "predicted_final_Marks": round(prediction, 2),
            "remark": remark
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
