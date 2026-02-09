from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import ee
import datetime
import json
import os 

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
}

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
        return "excess moisture in soil"

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

    crop = data.crop or "general"

    ndvi_val = round(stats["NDVI"], 3)
    ndmi_val = round(stats["NDMI"], 3)

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
        "map_image_url": generate_ndvi_thumbnail(ndvi, geometry)
    }
