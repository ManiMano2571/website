from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from PIL import Image
import io
import pickle
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins (or restrict as necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; modify for tighter security if necessary
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the models
def load_model(filepath):
    try:
        print(f"Attempting to load model: {filepath}")
        if filepath.endswith(".pkl"):
            # For CPU-only deployment, no CUDA-related deserialization
            return pickle.load(open(filepath, "rb"))
        elif filepath.endswith(".pt") or filepath.endswith(".pth"):
            # Explicitly map to CPU
            return torch.load(filepath, map_location=torch.device('cpu'))
        else:
            raise ValueError("Unsupported model format.")
    except Exception as e:
        print(f"Error loading model {filepath}: {str(e)}")
        return None


models = {
    "brain": load_model("Brain_Tumor_Detectionr_model.pkl"),
    "tb": load_model("tb_detector_model.pkl"),
    "pneumonia": load_model("chest-xray-pneumonia_detector_model.pkl")
}

# Validate that all models loaded successfully
if any(model is None for model in models.values()):
    raise RuntimeError("One or more models failed to load. Check file paths.")

@app.post("/")
async def diagnose(model: str = Form(...), image: UploadFile = None):
    if model not in models:
        return JSONResponse({"error": "Invalid model selected"}, status_code=400)

    if not image:
        return JSONResponse({"error": "No image provided"}, status_code=400)

    try:
        # Preprocess the image
        image_data = Image.open(io.BytesIO(await image.read()))
        image_data = image_data.convert("RGB").resize((224, 224))  # Convert to RGB and resize
        image_array = np.array(image_data) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Perform inference
        selected_model = models[model]
        prediction = selected_model.predict(image_array)
        diagnosis = "Positive" if prediction[0] == 1 else "Negative"
        return {"diagnosis": diagnosis}
    except Exception as e:
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=500)
