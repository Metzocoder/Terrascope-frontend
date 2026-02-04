from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import ee
import datetime
import easyocr
import re
from pdf2image import convert_from_path
import os

# ======================================
# INITIALIZE GOOGLE EARTH ENGINE
# ======================================
ee.Initialize(project="dotted-empire-477317-b8")

# ======================================
# FASTAPI SETUP
# ======================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# OCR INITIALIZATION
# ======================================
ocr_reader = easyocr.Reader(['mr', 'en'], gpu=False)

def run_ocr(image_path: str):
    """
    Runs OCR on an image and returns results
    """
    return ocr_reader.readtext(image_path)

# ======================================
# REQUEST MODELS
# ======================================
class PointData(BaseModel):
    lat: float
    lon: float

class FieldRequest(BaseModel):
    point: Optional[PointData] = None
    polygon: Optional[List[List[List[float]]]] = None
    crop: Optional[str] = "general"

# ======================================
# CLASSIFICATION HELPERS
# ======================================
def classify_ndvi(v: float) -> str:
    if v < 0.25:
        return "Low"
    elif v < 0.5:
        return "Moderate"
    else:
        return "High"

def classify_ndmi(v: float) -> str:
    if v < 0.05:
        return "Low"
    elif v < 0.25:
        return "Moderate"
    else:
        return "High"

# ======================================
# CROP-SPECIFIC THRESHOLDS
# ======================================

CROP_CONFIG = {
    "general": {
        "ndvi": {
            "very_poor": 0.25,
            "poor": 0.40,
            "good": 0.60
        },
        "ndmi": {
            "very_dry": 0.08,
            "dry": 0.18,
            "optimal": 0.30
        }
    },

    "cotton": {
        "ndvi": {
            "very_poor": 0.28,
            "poor": 0.45,
            "good": 0.65
        },
        "ndmi": {
            "very_dry": 0.10,
            "dry": 0.22,
            "optimal": 0.35
        }
    },

    "rice": {
        "ndvi": {
            "very_poor": 0.35,
            "poor": 0.55,
            "good": 0.75
        },
        "ndmi": {
            "very_dry": 0.25,
            "dry": 0.40,
            "optimal": 0.55
        }
    },

    "wheat": {
        "ndvi": {
            "very_poor": 0.30,
            "poor": 0.48,
            "good": 0.65
        },
        "ndmi": {
            "very_dry": 0.12,
            "dry": 0.25,
            "optimal": 0.38
        }
    }
}


# ======================================
# HUMAN READABLE INTERPRETATION
# ======================================

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
        return "excess moisture in soil"


# ======================================
# NDVI HEATMAP THUMBNAIL
# ======================================
def generate_ndvi_thumbnail(ndvi_image, geometry):
    vis_params = {
        "min": 0.2,
        "max": 0.8,
        "palette": [
            "8b0000",  # dark red
            "ff4500",  # orange red
            "ffd700",  # yellow
            "7fff00",  # light green
            "006400"   # dark green
        ]
    }

    return ndvi_image.visualize(**vis_params).getThumbURL({
        "region": geometry,
        "dimensions": 512,
        "format": "png"
    })


# ======================================
# SATBARA OCR PARSER (FINAL)
# ======================================
def extract_satbara_data(ocr_results):
    data = {
        "survey_number": None,
        "area": None
    }

    # keep only usable OCR lines
    lines = [(text.strip(), conf) for _, text, conf in ocr_results if conf > 0.25]

    # -------- SURVEY / GAT NUMBER --------
    for i, (text, conf) in enumerate(lines):
        if any(k in text for k in ["क्रमां", "उपविभाग", "गट", "उपविभाग"]):
            context = " ".join(
                lines[j][0] for j in range(i, min(i + 3, len(lines)))
            )

            nums = re.findall(r"\d{1,4}", context)
            if nums:
                data["survey_number"] = nums[0]
                break

    # -------- AREA (SATBARA FORMAT SAFE) --------
    area_candidates = []

    for i, (text, conf) in enumerate(lines):
        if any(k in text for k in ["हे", "हेक", "चौमी", "घार", "हिःघर"]):
            context = " ".join(
                lines[j][0] for j in range(max(0, i - 2), min(i + 3, len(lines)))
            )

            # Match formats:
            # 0.04.00 , 0.21.00 , 1.10 , 0.60
            nums = re.findall(r"\d+\.\d+(?:\.\d+)?", context)

            for n in nums:
                try:
                    parts = n.split(".")

                    # Convert Satbara style (X.YY.ZZ → hectares approx)
                    if len(parts) == 3:
                        val = float(parts[0]) + float(parts[1]) / 100
                    else:
                        val = float(n)

                    # STRICT realistic limits
                    if 0.01 <= val <= 50:
                        area_candidates.append((val, conf))

                except:
                    pass

    # choose highest confidence area
    if area_candidates:
        area_candidates.sort(key=lambda x: x[1], reverse=True)
        data["area"] = f"{area_candidates[0][0]:.2f}"

    return data


# ======================================
# SATBARA OCR ENDPOINT
# ======================================
@app.post("/satbara/ocr")
async def satbara_ocr(file: UploadFile = File(...)):

    os.makedirs("temp", exist_ok=True)

    safe_filename = file.filename.replace(" ", "_")
    temp_path = os.path.join("temp", safe_filename)

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # -------- PDF OR IMAGE --------
    if safe_filename.lower().endswith(".pdf"):
        images = convert_from_path(temp_path, dpi=300)
        image_path = os.path.join("temp", "satbara_page.png")
        images[0].save(image_path, "PNG")
    else:
        image_path = temp_path

    # -------- OCR --------
    ocr_results = run_ocr(image_path)

    # -------- DEBUG --------
    print("\n========== OCR RESULTS ==========")
    for _, text, conf in ocr_results:
        print(f"{text} | confidence: {conf}")
    print("================================\n")

    extracted = extract_satbara_data(ocr_results)

    return {
    "status": "needs_boundary",
    "extracted": {
        "survey_number": extracted.get("survey_number"),
        "area": extracted.get("area")
    }
}


class LandSaveRequest(BaseModel):
    survey_number: Optional[str]
    area: Optional[str]
    geometry: dict   # GeoJSON from Leaflet


@app.post("/lands")
def save_land(data: LandSaveRequest):
    """
    Saves user-drawn land boundary + Satbara metadata
    """

    # 🔴 For now we just return it (DB later)
    return {
        "status": "saved",
        "survey_number": data.survey_number,
        "area": data.area,
        "geometry_type": data.geometry.get("type")
    }


# ======================================
# MAIN ANALYSIS ENDPOINT (UNCHANGED)
# ======================================
@app.post("/analyze-field")
def analyze_field(data: FieldRequest):

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

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=30)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(geometry)
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")

    stats = ee.Image.cat([ndvi, ndmi]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    ).getInfo()

    if stats.get("NDVI") is None:
        return {"error": "No usable satellite data"}

    # ✅ CROP-SPECIFIC INTERPRETATION (FIXED POSITION)
    crop = data.crop or "general"

    ndvi_val = round(stats["NDVI"], 3)
    ndmi_val = round(stats["NDMI"], 3)

    # 🔥 GENERATE NDVI HEATMAP IMAGE
    ndvi_map_url = generate_ndvi_thumbnail(ndvi, geometry)

    return {
        "geometry_type": geometry_type,

        "ndvi": {
            "value": ndvi_val,
            "meaning": interpret_ndvi(ndvi_val, crop)
        },

        "ndmi": {
            "value": ndmi_val,
            "meaning": interpret_ndmi(ndmi_val, crop)
        },

        "crop": crop,
        "analysis_window_days": 30,

        # 🟢 THIS IS WHAT YOUR FRONTEND NEEDS
        "map_image_url": ndvi_map_url
    }



