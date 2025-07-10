# 🧠 Diabetes Prediction using IBM Watson AutoAI

This project uses IBM Watson Studio's **AutoAI** to develop a machine learning model that predicts the likelihood of diabetes based on patient data. It automates the full pipeline — from data preprocessing to model selection and optimization — and deploys the best model using a secure API for real-time predictions in a Colab notebook.

---

## 📁 Dataset

- **Source**: `diabetes.csv`
- **Features**:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- **Target**:
  - `Outcome` (1 = diabetic, 0 = non-diabetic)

---

## 🎯 Objective

The goal is to:
- Build an accurate diabetes prediction model using **AutoAI**.
- Deploy the best pipeline (selected from 8 AutoAI-generated pipelines).
- Use the **deployment API** to perform predictions in a Google Colab notebook.

---

## ⚙️ Tools & Technologies

- IBM Watson Studio
- IBM AutoAI
- Python (for Colab integration)
- IBM Cloud Deployment API
- Pandas, Requests (for API calls)

---

## 🚀 Deployment

1. **Train & Optimize** models using AutoAI.
2. **Deploy** the best pipeline (e.g., P8) to Watson Machine Learning.
3. **Access the deployment endpoint** and get the API key.
4. In **Colab Notebook**:
   - Authenticate using IAM token service
   - Send POST requests to the deployment endpoint
   - Receive prediction response (Outcome: 0 or 1)

---

## 📊 Visual Summary

- **Progress Map**: Shows the AutoAI pipeline stages — model selection, hyperparameter tuning, and feature engineering.
- **Relationship Map**: Displays how datasets, algorithms, and transformers connect within the experiment.
- **Best Model**: Pipeline P8 using Snap Boosting Machine Classifier.

---

## 📝 Example Colab API Snippet

```python
import requests
import json

# Replace with your actual API key and endpoint
API_KEY = "your_ibm_cloud_api_key"
DEPLOYMENT_URL = "your_model_endpoint_url"
<img width="1762" height="703" alt="prediction result" src="https://github.com/user-attachments/assets/176d558c-b8ff-47b5-8eeb-5ba4fabbff38" />
<img width="1907" height="757" alt="Relationship map" src="https://github.com/user-attachments/assets/b4a635c5-8b8b-42b7-a0e0-5c9006fb65d5" />
<img width="1500" height="682" alt="progress pipeline map" src="https://github.com/user-attachments/assets/c547157a-0a71-4b4e-b184-a74296bd9217" />

# Step 1: Get IAM token
token_response = requests.post(
    'https://iam.cloud.ibm.com/identity/token',
    data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
)
ml_token = token_response.json()["access_token"]

# Step 2: Prepare input data
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ml_token}"}
payload = {
    "input_data": [{
        <img width="1467" height="793" alt="api reference " src="https://github.com/user-attachments/assets/e4a89486-358a-410c-88fc-be57c2cf7b54" />
"fields": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "values": [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
    }]
}<img width="1800" height="782" alt="collab deployment" src="https://github.com/user-attachments/assets/58dd8ad5-027c-4305-8827-6f9e484cc656" />
<img width="1553" height="853" alt="deployment status" src="https://github.com/user-attachments/assets/17fbd39e-9a4e-4a04-9a7b-374896264bac" />
<img width="753" height="503" alt="dataset preview" src="https://github.com/user-attachments/assets/cce14e0e-d831-4dae-a878-9a15b0e4fd1e" />

<img width="1883" height="637" alt="testing" src="https://github.com/user-attachments/assets/76916bde-7fa0-469b-9da5-eab71bc77c16" />

# Step 3: Send request and get prediction
response = requests.post(DEPLOYMENT_URL, json=payload, headers=headers)
print("Prediction:", response.json())
