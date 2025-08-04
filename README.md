# 📚 Student Performance Prediction 

A machine learning-powered web API using **FastAPI** to predict the **final grade (G3)** of students based on demographic, academic, family, and personal data. This project is based on the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

---

## 🚀 Features

- Predicts final student grade on a 0–20 scale.
- Organized input via four intuitive sections:
  - Student Details
  - Family Background
  - Academic Status
  - Health Status
- Categorical label encoding with automatic casing normalization.
- Grading remark generation (A+, B, C, etc.).
- Easy Flutter/mobile app integration.

---

## 📂 Project Structure

```
student_performance_prediction/
├── app.py                                            # FastAPI backend logic
├── Student_Performance_model.py                      # Model training script
├── student_performance_portuguese_dataset.csv        # Cleaned Portuguese dataset used for training
├── student_performance_model.joblib                  # Trained RandomForest model
├── label_encoders.joblib                             # Encoders for categorical columns
├── requirements.txt                                  # Python package dependencies
└── README.md                                         # Project documentation
```

---

## 🛠️ Installation

### Prerequisites:
- Python 3.9.5

### Setup:
```bash
# Repository Name
Student_Performance_Prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate 

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Model Training

To train or retrain the model using the dataset:

```bash
python Student_Performance_model.py
```

This will generate the model and encoder files:
- `student_performance_model.joblib`
- `label_encoders.joblib`

---

## 🚦 Running the API

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

Navigate to:
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Render: [https://student-performance-prediction-hh3v.onrender.com](https://student-performance-prediction-hh3v.onrender.com)

---

## 📥 API Usage

### Endpoint:
```
POST /predict
```

### Request Body Example:
```json
{
  "student": {
    "school": "GP",
    "sex": "M",
    "age": 17,
    "address": "U"
  },
  "family": {
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 2,
    "Mjob": "teacher",
    "Fjob": "services",
    "reason": "reputation",
    "guardian": "mother",
    "famsup": "yes"
  },
  "academic": {
    "traveltime": 1,
    "studytime": 3,
    "failures": 0,
    "schoolsup": "no",
    "paid": "yes",
    "activities": "yes",
    "internet": "yes",
    "nursery": "yes",
    "higher": "yes",
    "absences": 4,
    "G1": 16,
    "G2": 18
  },
  "health": {
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 1,
    "health": 4
  }
}
```

### Sample Response:
```json
{
  "predicted_final_Marks": 17.3,
  "remark": "A (Very Good)"
}
```

---

## 🧠 Model Overview

- Algorithm: `RandomForestRegressor`
- Input Features: 31
- Target: `G3` (final grade)
- Evaluation: R² Score ~ 0.84

---

## 🖼️ Enhancement Ideas

- Show emoji/photo based on grade in frontend.
- Implement history tracking per student.
- Deploy via Render, Vercel, or Docker.

---

## 📘 Dataset Info

- Source: UCI Portuguese Student Dataset
- [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)

---

