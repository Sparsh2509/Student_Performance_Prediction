from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("student_performance_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# Define all 32 input features (except 'G3')
class StudentInput(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int
    G1: int
    G2: int

# FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the UCI Student Performance Prediction API!"}

@app.post("/predict")
def predict(data: StudentInput):
    try:
        input_data = data.dict()

        # Apply label encoders to categorical fields
        for col in label_encoders:
            encoder = label_encoders[col]
            if input_data[col] not in encoder.classes_:
                return {"error": f"Unknown category '{input_data[col]}' for field '{col}'."}
            input_data[col] = encoder.transform([input_data[col]])[0]

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])[model.feature_names_in_]

        # Predict
        prediction = model.predict(df_input)[0]
        return {"predicted_final_grade": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
