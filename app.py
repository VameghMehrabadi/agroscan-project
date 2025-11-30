from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from googletrans import Translator
import io
import sys

# ---------- 1. تنظیمات و مسیر فایل‌ها ----------
# مطمئن شو این فایل‌ها کنار app.py باشن
CLASSES_PATH = "./classes.txt"
CKPT_PATH = "./model.pth"
CSV_PATH = "./treatmentsdrugs.csv"

# تنظیم دستگاه (CPU یا GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. لود کردن لیست کلاس‌ها ----------
try:
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = f.read().splitlines()
    print(f"[INFO] {len(class_names)} classes loaded.")
except Exception as e:
    print(f"[ERROR] Could not load classes.txt: {e}")
    sys.exit(1) # اگر کلاس‌ها نباشن برنامه نباید اجرا شه

# ---------- 3. ساخت و لود مدل ----------
print("[INFO] Loading model...")
try:
    # ساختار مدل دقیقاً مثل کدی که فرستادی
    model = models.resnet18(weights=None) # یا pretrained=False در نسخه‌های قدیمی
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    
    # لود کردن وزن‌ها
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("[OK] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model.pth: {e}")
    # اینجا برنامه رو متوقف نمی‌کنیم ولی ای‌پی‌آی کار نخواهد کرد

# ---------- 4. لود کردن فایل اکسل/CSV درمان‌ها ----------
try:
    treatments_df = pd.read_csv(CSV_PATH)
    # حذف فاصله‌های اضافی احتمالی
    treatments_df['label'] = treatments_df['label'].astype(str).str.strip()
    treatments_df['region'] = treatments_df['region'].astype(str).str.strip()
    print("[OK] Treatments CSV loaded.")
except Exception as e:
    print(f"[WARNING] Could not load CSV file: {e}")
    treatments_df = pd.DataFrame() # یک دیتای خالی می‌سازیم که ارور نده

# مترجم
translator = Translator()

# پیش‌پردازش تصویر (Transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------- 5. تعریف API ----------
app = FastAPI(title="AgroScan API")

# تنظیمات CORS (برای اتصال به فرانت‌اند)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # در حالت پروداکشن بهتره آدرس سایتت رو بذاری
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
    lang: str = Form("fa") # زبان پیش‌فرض فارسی
):
    # الف: خواندن و پردازش تصویر
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # ب: پیش‌بینی با مدل
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        
        predicted_label = class_names[pred_idx.item()]
        confidence_score = confidence.item()

    # پ: پیدا کردن درمان در فایل CSV
    # فیلتر کردن بر اساس نام بیماری و منطقه
    row = treatments_df[
        (treatments_df['label'] == predicted_label) & 
        (treatments_df['region'] == region)
    ]

    if not row.empty:
        treatment_text = row.iloc[0]['treatment']
    else:
        treatment_text = "No treatment found for this disease in the selected region."

    # ت: ترجمه متن (اگر زبان فارسی نباشد یا متن نیاز به ترجمه داشته باشد)
    final_treatment = treatment_text
    
    # اگر زبان درخواستی با زبان دیتابیس فرق داره ترجمه کن (اینجا فرض کردیم دیتابیس انگلیسی یا فارسیه)
    if lang and lang != "fa": 
        try:
            translated = translator.translate(treatment_text, dest=lang)
            final_treatment = translated.text
        except Exception as e:
            print(f"[WARNING] Translation failed: {e}")
            # در صورت خطا در ترجمه، همون متن اصلی رو می‌فرستیم

    return {
        "prediction": predicted_label,
        "region": region,
        "confidence": round(confidence_score * 100, 2),
        "treatment": final_treatment,
        "original_treatment": treatment_text
    }

# این قسمت باعث میشه با دستور python app.py اجرا بشه
if __name__ == "__main__":
    import uvicorn
    print("[INFO] Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)