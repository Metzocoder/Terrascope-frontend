from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import ee
import datetime
import json
import os 

# =====================================================
# NIR MODEL LOADING
# =====================================================
import joblib
import numpy as np
import cv2
from fastapi import UploadFile, File

NIR_MODEL = None
NIR_SCALER = None

try:
    NIR_MODEL = joblib.load("models/terrascope_nir_model.pkl")
    NIR_SCALER = joblib.load("models/terrascope_nir_scaler.pkl")
    print("🟢 NIR model loaded successfully")
except Exception as e:
    print("🔴 NIR model loading failed:", str(e))

# =====================================================
# GOOGLE EARTH ENGINE INITIALIZATION (SERVICE ACCOUNT)
# =====================================================
GEE_INITIALIZED = False
GEE_ERROR = None

try:
    service_account_info = json.loads(
        os.environ["GEE_SERVICE_ACCOUNT_JSON"]
    )

    credentials = ee.ServiceAccountCredentials(
        service_account_info["client_email"],
        key_data=json.dumps(service_account_info)
    )

    ee.Initialize(
        credentials=credentials,
        project=service_account_info["project_id"]
    )

    GEE_INITIALIZED = True
    print("🟢 GEE initialized using env credentials")

except Exception as e:
    GEE_ERROR = str(e)
    print("🔴 GEE init failed")
    print(GEE_ERROR)


# =====================================================
# FASTAPI SETUP
# =====================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gee_initialized": GEE_INITIALIZED,
        "gee_error": GEE_ERROR
    }

# =====================================================
# REQUEST MODELS
# =====================================================
class PointData(BaseModel):
    lat: float
    lon: float

class FieldRequest(BaseModel):
    point: Optional[PointData] = None
    polygon: Optional[List[List[List[float]]]] = None
    crop: Optional[str] = "general"

class LandDetails(BaseModel):
    survey_number: str
    area: float  # hectares

# =====================================================
# CROP-SPECIFIC CONFIG
# These thresholds define what's "poor" vs "good"
# for each specific crop type, based on agronomic data.
# =====================================================
CROP_CONFIG = {
    "general": {
        "ndvi": {"very_poor": 0.25, "poor": 0.40, "good": 0.60},
        "ndmi": {"very_dry": 0.08, "dry": 0.18, "optimal": 0.30},
    },
    "cotton": {
        "ndvi": {"very_poor": 0.28, "poor": 0.45, "good": 0.65},
        "ndmi": {"very_dry": 0.10, "dry": 0.22, "optimal": 0.35},
    },
    "rice": {
        "ndvi": {"very_poor": 0.35, "poor": 0.55, "good": 0.75},
        "ndmi": {"very_dry": 0.25, "dry": 0.40, "optimal": 0.55},
    },
    "wheat": {
        "ndvi": {"very_poor": 0.30, "poor": 0.48, "good": 0.65},
        "ndmi": {"very_dry": 0.12, "dry": 0.25, "optimal": 0.38},
    },
    "soybean": {
        "ndvi": {"very_poor": 0.28, "poor": 0.45, "good": 0.65},
        "ndmi": {"very_dry": 0.10, "dry": 0.20, "optimal": 0.35},
    },
    "sugarcane": {
        "ndvi": {"very_poor": 0.40, "poor": 0.55, "good": 0.70},
        "ndmi": {"very_dry": 0.15, "dry": 0.28, "optimal": 0.42},
    },
}

# =====================================================
# SCORE HELPERS
# Converts a raw NDVI/NDMI value to a 0–100 score
# using crop-specific thresholds for calibration.
# This keeps scores consistent between frontend and backend.
# =====================================================
def ndvi_to_score(ndvi: float, crop: str) -> int:
    """
    Maps NDVI to 0–100 using crop-specific thresholds.
    - very_poor thresh → score of ~5
    - good thresh       → score of ~85
    - Linearly interpolated, clamped to [0, 100]
    """
    cfg = CROP_CONFIG.get(crop, CROP_CONFIG["general"])["ndvi"]
    low = cfg["very_poor"]  # ~5 score
    high = cfg["good"]      # ~85 score
    score = ((ndvi - low) / (high - low)) * 80 + 5
    return max(0, min(100, round(score)))

def ndmi_to_score(ndmi: float, crop: str) -> int:
    """
    Maps NDMI to 0–100 using crop-specific thresholds.
    """
    cfg = CROP_CONFIG.get(crop, CROP_CONFIG["general"])["ndmi"]
    low = cfg["very_dry"]
    high = cfg["optimal"]
    score = ((ndmi - low) / (high - low)) * 80 + 5
    return max(0, min(100, round(score)))


# =====================================================
# INTERPRETATION HELPERS
# =====================================================
def interpret_ndvi(ndvi: float, crop: str):
    cfg = CROP_CONFIG.get(crop, CROP_CONFIG["general"])["ndvi"]
    if ndvi < cfg["very_poor"]:
        return "very weak crop growth"
    elif ndvi < cfg["poor"]:
        return "weak crop growth"
    elif ndvi < cfg["good"]:
        return "fairly healthy crop growth"
    else:
        return "very healthy and dense crop growth"

def interpret_ndmi(ndmi: float, crop: str):
    cfg = CROP_CONFIG.get(crop, CROP_CONFIG["general"])["ndmi"]
    if ndmi < cfg["very_dry"]:
        return "severe water stress"
    elif ndmi < cfg["dry"]:
        return "low moisture in soil"
    elif ndmi < cfg["optimal"]:
        return "adequate soil moisture"
    else:
        return "good soil moisture levels"

# =====================================================
# NIR FEATURE EXTRACTION (7 FEATURES)
# =====================================================
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

def extract_nir_features(img):

    img = cv2.resize(img, (224, 224))

    # CLAHE normalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    b, g, r = cv2.split(img)

    # Vegetation Index
    vi = (r - g) / (r + g + 1e-5)
    vi_mean = float(np.mean(vi))
    vi_std = float(np.std(vi))

    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    edges = img_as_ubyte(edges)
    glcm = graycomatrix(
        edges,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = float(graycoprops(glcm, 'contrast')[0][0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0][0])
    energy = float(graycoprops(glcm, 'energy')[0][0])
    entropy = float(-np.sum(glcm * np.log2(glcm + 1e-10)))

    edge_density = float(np.sum(edges > 0) / (224 * 224))

    return np.array([[
        vi_mean,
        vi_std,
        contrast,
        homogeneity,
        energy,
        entropy,
        edge_density
    ]])

# =====================================================
# NDVI MAP THUMBNAIL
# =====================================================
def generate_ndvi_thumbnail(ndvi_image, geometry):
    return ndvi_image.visualize(
        min=0.2,
        max=0.8,
        palette=["8b0000", "ff4500", "ffd700", "7fff00", "006400"]
    ).getThumbURL({
        "region": geometry,
        "dimensions": 512,
        "format": "png"
    })

# =====================================================
# SMART IMAGE COLLECTION BUILDER
# Uses a tiered fallback strategy:
#   1. Try 15 days, cloud < 20%
#   2. Try 30 days, cloud < 30%
#   3. Try 60 days, cloud < 50%
# This prevents empty results in cloudy regions.
# =====================================================
def get_best_collection(geometry):
    today = datetime.date.today()
    strategies = [
        (15, 20),  # Preferred: recent, clean
        (30, 30),  # Fallback 1: slightly wider window
        (60, 50),  # Fallback 2: longer window for very cloudy areas
    ]

    for days, cloud_pct in strategies:
        start = today - datetime.timedelta(days=days)
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(str(start), str(today))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        )
        count = collection.size().getInfo()
        if count > 0:
            print(f"✅ Found {count} images with window={days}d, cloud<{cloud_pct}%")
            return collection.median().clip(geometry), days, cloud_pct

    return None, None, None

# =====================================================
# LAND METADATA (MANUAL INPUT)
# =====================================================
@app.post("/lands")
def save_land(data: LandDetails):
    return {
        "status": "saved",
        "survey_number": data.survey_number,
        "area": data.area
    }

# =====================================================
# MAIN ANALYSIS ENDPOINT
# =====================================================
@app.post("/analyze-field")
def analyze_field(data: FieldRequest):

    if not GEE_INITIALIZED:
        return {
            "error": "Google Earth Engine not initialized",
            "details": GEE_ERROR
        }

    if data.polygon:
        geometry = ee.Geometry.Polygon(data.polygon)
        geometry_type = "polygon"

    elif data.point:
        geometry = ee.Geometry.Point(
            [data.point.lon, data.point.lat]
        ).buffer(1000)
        geometry_type = "point"

    else:
        return {"error": "No geometry provided"}

    # Use smart fallback collection builder
    s2, window_days, cloud_pct_used = get_best_collection(geometry)

    if s2 is None:
        return {
            "error": "No usable satellite data found even after fallback. Area may have persistent cloud cover."
        }

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")

    stats = ee.Image.cat([ndvi, ndmi]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    ).getInfo()

    if stats.get("NDVI") is None or stats.get("NDMI") is None:
        return {"error": "No usable satellite data after analysis"}

    crop = data.crop or "general"

    ndvi_val = round(stats["NDVI"], 3)
    ndmi_val = round(stats["NDMI"], 3)

    # Compute crop-aware scores (0–100)
    crop_score = ndvi_to_score(ndvi_val, crop)
    water_score = ndmi_to_score(ndmi_val, crop)

    return {
        "geometry_type": geometry_type,
        "ndvi": {
            "value": ndvi_val,
            "score": crop_score,           # 0-100, crop-calibrated
            "meaning": interpret_ndvi(ndvi_val, crop)
        },
        "ndmi": {
            "value": ndmi_val,
            "score": water_score,          # 0-100, crop-calibrated
            "meaning": interpret_ndmi(ndmi_val, crop)
        },
        "crop": crop,
        "analysis_window_days": window_days,
        "cloud_threshold_used": cloud_pct_used,
        "map_image_url": generate_ndvi_thumbnail(ndvi, geometry)
    }

# =====================================================
# GROUND NIR IMAGE ANALYSIS
# =====================================================
@app.post("/analyze-nir-image")
async def analyze_nir_image(file: UploadFile = File(...)):

    if NIR_MODEL is None or NIR_SCALER is None:
        return {"error": "NIR model not loaded"}

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    features = extract_nir_features(img)
    
    # ── VALIDATION HEURISTIC ──
    # features[0][0] is vi_mean (Vegetation Index)
    # features[0][6] is edge_density
    vi_mean = features[0][0]
    edge_density = features[0][6]
    
    # Heuristic: Plant images typically have high VI (green contrast) 
    # and a certain range of edge density (organic structure).
    # Attendance photos/faces/rooms usually have low VI or very different texture.
    is_plant = (vi_mean > -0.05) and (edge_density > 0.01)
    
    if not is_plant:
        return {
            "error": "invalid_specimen",
            "message": "The image does not look like a valid plant specimen. Please ensure the leaf is clearly visible and well-lit."
        }

    features_scaled = NIR_SCALER.transform(features)

    prob = float(NIR_MODEL.predict_proba(features_scaled)[0][1])
    stress_score = round(prob * 100, 2)

    if stress_score < 30:
        level = "Healthy"
        recommendation = "No immediate action required."
    elif stress_score < 55:
        level = "Mild Stress"
        recommendation = "Monitor crop condition regularly."
    elif stress_score < 75:
        level = "Moderate Stress"
        recommendation = "Inspect irrigation and nutrient supply."
    else:
        level = "Severe Stress"
        recommendation = "Immediate field inspection recommended."

    return {
        "stress_probability": prob,
        "stress_score": stress_score,
        "stress_level": level,
        "recommendation": recommendation
    }

    print("NIR endpoint registered successfully")