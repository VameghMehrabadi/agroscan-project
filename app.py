from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import gdown
import os
import sys
import io

# ---------- فایل‌ها و مسیرها ----------
CLASSES_PATH = "./classes.txt"
CSV_PATH = "./treatmentsdrugs.csv"
MODEL_PATH = "./model.pth"

# لینک Google Drive مدل
FILE_ID = "1cmL7eJV7UEhTGN9M0F6yKJzOwJy90_3V"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ---------- تنظیم دستگاه ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- دانلود مدل اگر وجود نداشت ----------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading model from Google Drive...")
        try:
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
            print("[OK] Model downloaded successfully")
        except Exception as e:
            print(f"[ERROR] Model download failed: {e}")
            sys.exit(1)

download_model()

# ---------- لود کلاس‌ها ----------
try:
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = f.read().splitlines()
    print(f"[OK] {len(class_names)} classes loaded")
except Exception as e:
    print(f"[ERROR] Could not load {CLASSES_PATH}: {e}")
    sys.exit(1)

# ---------- ساخت و لود مدل ----------
print("[INFO] Loading model...")
try:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("[OK] Model loaded and ready")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# ---------- لود CSV درمان‌ها ----------
try:
    treatments_df = pd.read_csv(CSV_PATH)
    treatments_df['label'] = treatments_df['label'].astype(str).str.strip()
    treatments_df['region'] = treatments_df['region'].astype(str).str.strip()
    print("[OK] Treatments CSV loaded")
except Exception as e:
    print(f"[WARNING] Could not load CSV file: {e}")
    treatments_df = pd.DataFrame()

# ---------- تنظیم transform تصویر ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------- ساخت اپ FastAPI ----------
app = FastAPI(title="AgroScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running", "device": str(device)}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    region: str = Form(...),
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
        label = class_names[idx.item()]
        confidence = round(conf.item() * 100, 2)

    row = treatments_df[
        (treatments_df['label'] == label) &
        (treatments_df['region'] == region)
    ]

    treatment = (
        row.iloc[0]['treatment']
        if not row.empty
        else "No treatment found for this disease and region."
    )

    return {
        "prediction": label,
        "confidence": confidence,
        "region": region,
        "treatment": treatment,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
