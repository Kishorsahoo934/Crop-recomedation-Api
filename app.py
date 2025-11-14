import os
import io
import re
import json
import pickle
import joblib
import uvicorn
import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import wikipedia

# ========== GLOBAL CONFIG ==========
genai.configure(api_key="AIzaSyBffMhdDvvjSeii4vsTy93jCHcBg81iz4o")

# System-level instruction for the chatbot and recommendation generation.
# This prompt instructs the model to return short, point-wise answers.
SYSTEM_PROMPT = (
    "You are FarmSathi, a friendly agricultural assistant who speaks in simple, easy-to-understand language. "
    "Format your responses like this:\n\n"
        "IMPORTANT: Only provide information about crops that are suitable for and commonly grown in Odisha state. "
        "Focus on local farming practices, climate conditions, and crop varieties specific to Odisha.\n\n"
        "For lists of crops (specific to Odisha):\n"
    "1. First item\n"
    "2. Second item\n"
    "3. Third item\n\n"
    "For explanations and steps:\n"
    "• Use bullet points\n"
    "• Write in simple words that farmers understand easily\n"
    "• Avoid technical terms - explain them if needed\n"
    "• Keep each point short (1-2 simple sentences)\n"
    "• Put each point on a new line\n\n"
    "Reference: Corn Common Rust (use this when the disease is Corn Common Rust)\n"
    "- Cause: Fungus (Puccinia sorghi) that prefers cool, wet, humid conditions.\n"
    "- Symptoms: Small raised orange-brown pustules on top/bottom of leaves; may release rusty spores when rubbed.\n"
    "- Why it matters: Damages leaves, reducing photosynthesis and yield.\n\n"
    "Short Treatment & Prevention (point-wise):\n"
    "1. Choose rust-resistant corn varieties when possible.\n"
    "2. Remove or bury crop debris after harvest; manage weeds to reduce disease reservoirs.\n"
    "3. Improve airflow: avoid dense planting; space rows to reduce humidity.\n"
    "4. If severe, apply an appropriate fungicide early; follow label instructions and safety precautions.\n"
    "5. Consult local extension services for region-specific fungicide recommendations.\n\n"
    "Always keep answers short, actionable, and in point form. If the user asks for more details, offer a short summary and suggest next steps."
)

app = FastAPI(
    title="🌾 FarmSathi Unified Agriculture API",
    description="One API for Crop Recommendation, Fertilizer, Statistics, Chatbot, and Disease Detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 1️⃣ CROP RECOMMENDATION ENDPOINT
# ======================================================
try:
    with open('crop_Recommendation_random_forest.pkl', 'rb') as file:
        crop_model = pickle.load(file)
    print("✅ Crop recommendation model loaded.")
except Exception as e:
    print(f"❌ Error loading crop model: {e}")
    crop_model = None

@app.post("/crop-recommend")
async def crop_recommend(
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...)
):
    if crop_model is None:
        raise HTTPException(status_code=500, detail="Crop model not loaded properly.")

    input_data = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    prediction = crop_model.predict(input_data)
    return {"recommended_crop": prediction[0]}


# ======================================================
# 2️⃣ CROP STATISTICS ENDPOINT
# ======================================================
# DATA_PATH = r'cropdiseaseprediction_final/Crop_Recommendation_Dataset.csv'
# SUMM_PATH = r'K:\testing\farmsathi\docs\crop_summaries.txt'

# try:
#     data = pd.read_csv(DATA_PATH)
#     crop_summaries = {}
#     with open(SUMM_PATH, 'r') as f:
#         for line in f:
#             crop, summary = line.split(': ', 1)
#             crop_summaries[crop] = summary.strip()
#     print("✅ Crop data and summaries loaded.")
# except Exception as e:
#     print(f"❌ Error loading dataset: {e}")
#     data, crop_summaries = None, {}

# @app.get("/crop-stats/{crop_name}")
# async def crop_stats(crop_name: str):
#     if data is None:
#         raise HTTPException(status_code=500, detail="Dataset not available.")
#     if crop_name not in data['Crop'].unique():
#         raise HTTPException(status_code=404, detail="Crop not found.")

#     x = data[data['Crop'] == crop_name]
#     stats = {
#         "Parameter": ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'],
#         "Minimum": [
#             x['Nitrogen'].min(), x['Phosphorus'].min(), x['Potassium'].min(),
#             x['Temperature'].min(), x['Humidity'].min(), x['pH_Value'].min(), x['Rainfall'].min()
#         ],
#         "Average": [
#             x['Nitrogen'].mean(), x['Phosphorus'].mean(), x['Potassium'].mean(),
#             x['Temperature'].mean(), x['Humidity'].mean(), x['pH_Value'].mean(), x['Rainfall'].mean()
#         ],
#         "Maximum": [
#             x['Nitrogen'].max(), x['Phosphorus'].max(), x['Potassium'].max(),
#             x['Temperature'].max(), x['Humidity'].max(), x['pH_Value'].max(), x['Rainfall'].max()
#         ]
#     }
#     summary = crop_summaries.get(crop_name, "No summary available for this crop.")
#     return {"statistics": stats, "summary": summary}


# # ======================================================
# # 3️⃣ FERTILIZER RECOMMENDATION ENDPOINT
# ======================================================
FERT_MODEL_PATH = 'fertilizer_recommendation_model_latest.joblib'
FERT_META_PATH = 'fertilizer_model_metadata_latest.json'

try:
    fert_model = joblib.load(FERT_MODEL_PATH)
    with open(FERT_META_PATH, 'r') as f:
        fert_metadata = json.load(f)
    FERT_FEATURES = fert_metadata['feature_info']['feature_columns']
    print("✅ Fertilizer model loaded.")
except Exception as e:
    print(f"❌ Error loading fertilizer model: {e}")
    fert_model, FERT_FEATURES = None, []

FERTILIZER_MAP = {
    0: '14-35-14', 1: '28-28', 2: 'DAP', 3: 'MOP', 4: 'Potash', 5: 'SSP', 6: 'Urea'
}
SOIL_TYPES = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
CROP_TYPES = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 
              'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

@app.post("/fertilizer-recommend")
async def fertilizer_recommend(
    temp: float = Form(...),
    humidity: float = Form(...),
    moisture: float = Form(...),
    nitrogen: float = Form(...),
    phosphorous: float = Form(...),
    potassium: float = Form(...),
    ph: float = Form(...),
    soil_type: str = Form(...),
    crop_type: str = Form(...)
):
    if fert_model is None:
        raise HTTPException(status_code=500, detail="Fertilizer model not available.")

    soil_encoded = SOIL_TYPES.index(soil_type)
    crop_encoded = CROP_TYPES.index(crop_type)
    df = pd.DataFrame([{
        'Temparature': temp,
        'Moisture': moisture,
        'Soil Type': soil_encoded,
        'Crop Type': crop_encoded,
        'Nitrogen': nitrogen,
        'Phosphorous': phosphorous,
        'Potassium': potassium,
        'pH': ph,
        'Humidity ': humidity
    }])[FERT_FEATURES]

    pred = fert_model.predict(df)[0]
    fertilizer = FERTILIZER_MAP.get(pred, "Unknown Fertilizer")
    return {"recommended_fertilizer": fertilizer}


# ======================================================
# 4️⃣ CHATBOT ENDPOINT (RAG + GEMINI)
# ======================================================
@app.post("/chatbot")
async def chatbot(query: str = Form(...)):
    """Gemini-based agricultural assistant."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        # Prepend the system-level instructions to ensure short, point-wise replies
        prompt = SYSTEM_PROMPT + "\n\nUser: " + query

        # Run synchronous API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt)
        )
        return {"response": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {e}")


# ======================================================
# 5️⃣ PLANT DISEASE DETECTION ENDPOINT
# ======================================================
MODEL_PATH = r"model.tflite"
CLASSES_PATH = r"class_indices.json"
IMAGE_SIZE = (224, 224)

# Initialize Gemini model once
disease_model = genai.GenerativeModel("models/gemini-2.5-flash")

try:
    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ Disease model loaded.")
except Exception as e:
    print(f"❌ Error loading disease model: {e}")
    interpreter, idx_to_class = None, None

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_disease(interpreter, input_array, idx_to_class):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(preds))
    return idx_to_class.get(idx, "Unknown"), float(preds[idx]) * 100

@app.post("/predict-disease")
async def predict_disease_api(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly.")
    try:
        # Validate file type
        content_type = getattr(file, 'content_type', '')
        if not content_type or not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Process image and get prediction
        input_arr = preprocess_image(image)
        disease, conf = predict_disease(interpreter, input_arr, idx_to_class)

        # Get recommendation with timeout
        recommendation = "No recommendation available"
        try:
            # Run Gemini API call with timeout
            # Prepend the system prompt so the reply is short and point-wise.
            prompt = SYSTEM_PROMPT + f"\n\nUser: Give simple treatment steps for {disease} in farmer-friendly language. Use bullet points and put each step on a new line."
            model = genai.GenerativeModel("gemini-2.5-flash")

            # Run synchronous API call in a thread pool to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(prompt)
            )
            recommendation = response.text.strip()
        except asyncio.TimeoutError:
            recommendation = "Recommendation generation timed out. Please try again."
        except Exception as e:
            recommendation = f"Could not generate recommendation: {str(e)}"

        return {
            "predicted_disease": disease,
            "confidence": f"{conf:.2f}",
            "recommendation": recommendation
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================
# ✅ RUN SERVER
# ======================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
