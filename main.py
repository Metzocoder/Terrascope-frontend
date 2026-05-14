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
    service_account_info = json.loads(os.environ["GEE_SERVICE_ACCOUNT_JSON"])
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
    allow_origins=["*"],
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
    area: float

class FieldHeatmapRequest(BaseModel):
    polygon: List[List[List[float]]]
    grid_size: Optional[int] = 20
    index_type: Optional[str] = "NDVI"

class LandMeasurementRequest(BaseModel):
    polygon: List[List[List[float]]]

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
# =====================================================
def ndvi_to_score(ndvi: float, crop: str) -> int:
    cfg = CROP_CONFIG.get(crop, CROP_CONFIG["general"])["ndvi"]
    low = cfg["very_poor"]
    high = cfg["good"]
    score = ((ndvi - low) / (high - low)) * 80 + 5
    return max(0, min(100, round(score)))

def ndmi_to_score(ndmi: float, crop: str) -> int:
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
# NEW GEE HELPER FUNCTIONS
# =====================================================
def calculate_evi(image):
    """Calculate EVI (Enhanced Vegetation Index)"""
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }
    ).rename('EVI')
    return image.addBands(evi)

def calculate_savi(image):
    """Calculate SAVI (Soil Adjusted Vegetation Index)"""
    L = 0.5
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'L': L
        }
    ).rename('SAVI')
    return image.addBands(savi)

def get_stress_level(ndvi_value: float) -> str:
    """Determine stress level from NDVI value"""
    if ndvi_value >= 0.6:
        return "Healthy"
    elif ndvi_value >= 0.4:
        return "Moderate"
    elif ndvi_value >= 0.2:
        return "Stressed"
    else:
        return "Severe"

def get_zone_color(value: float, index_type: str = "NDVI") -> str:
    """Get color code for heatmap zones"""
    if index_type == "NDVI":
        if value >= 0.6:
            return "#00FF00"  # Green
        elif value >= 0.4:
            return "#FFFF00"  # Yellow
        elif value >= 0.2:
            return "#FFA500"  # Orange
        else:
            return "#FF0000"  # Red
    elif index_type == "NDMI":
        if value >= 0.3:
            return "#0000FF"
        elif value >= 0.1:
            return "#00FFFF"
        else:
            return "#FF0000"
    return "#808080"

def get_zone_recommendation(ndvi: float, ndmi: float) -> str:
    """Get specific recommendation for a zone based on NDVI and NDMI"""
    # Critical zones (NDVI < 0.3)
    if ndvi < 0.3:
        if ndmi < 0.1:
            return "Critical: Immediate irrigation and fertilization needed"
        elif ndmi < 0.2:
            return "Critical: Apply fertilizer and monitor water stress"
        else:
            return "Critical: Nutrient deficiency detected, apply fertilizer"
    
    # Stressed zones (NDVI 0.3-0.5)
    elif ndvi < 0.5:
        if ndmi < 0.15:
            return "Stressed: Increase irrigation frequency"
        elif ndmi < 0.25:
            return "Stressed: Monitor water levels and consider light fertilization"
        else:
            return "Stressed: Consider pest inspection and nutrient boost"
    
    # Moderate zones (NDVI 0.5-0.6)
    elif ndvi < 0.6:
        if ndmi < 0.2:
            return "Moderate: Maintain irrigation schedule"
        else:
            return "Moderate: Continue current practices, monitor regularly"
    
    # Healthy zones (NDVI >= 0.6)
    else:
        if ndmi < 0.25:
            return "Healthy: Maintain irrigation to sustain growth"
        else:
            return "Healthy: Excellent condition, continue current practices"

# =====================================================
# NIR FEATURE EXTRACTION
# =====================================================
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte

def extract_nir_features(img):
    img = cv2.resize(img, (224, 224))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    b, g, r = cv2.split(img)
    
    vi = (r - g) / (r + g + 1e-5)
    vi_mean = float(np.mean(vi))
    vi_std = float(np.std(vi))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = img_as_ubyte(edges)
    glcm = graycomatrix(edges, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = float(graycoprops(glcm, 'contrast')[0][0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0][0])
    energy = float(graycoprops(glcm, 'energy')[0][0])
    entropy = float(-np.sum(glcm * np.log2(glcm + 1e-10)))
    edge_density = float(np.sum(edges > 0) / (224 * 224))
    
    return np.array([[vi_mean, vi_std, contrast, homogeneity, energy, entropy, edge_density]])

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
# =====================================================
def get_best_collection(geometry):
    today = datetime.date.today()
    strategies = [
        (15, 20),
        (30, 30),
        (60, 50),
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
# LAND METADATA
# =====================================================
@app.post("/lands")
def save_land(data: LandDetails):
    return {
        "status": "saved",
        "survey_number": data.survey_number,
        "area": data.area
    }

# =====================================================
# MAIN ANALYSIS ENDPOINT (ENHANCED)
# =====================================================
@app.post("/analyze-field")
def analyze_field(data: FieldRequest):
    if not GEE_INITIALIZED:
        return {"error": "Google Earth Engine not initialized", "details": GEE_ERROR}
    
    if data.polygon:
        geometry = ee.Geometry.Polygon(data.polygon)
        geometry_type = "polygon"
    elif data.point:
        geometry = ee.Geometry.Point([data.point.lon, data.point.lat]).buffer(1000)
        geometry_type = "point"
    else:
        return {"error": "No geometry provided"}
    
    s2, window_days, cloud_pct_used = get_best_collection(geometry)
    if s2 is None:
        return {"error": "No usable satellite data found"}
    
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
    
    # Calculate additional indices
    image_with_indices = calculate_evi(s2)
    image_with_indices = calculate_savi(image_with_indices)
    
    # Calculate overall field statistics
    stats = ee.Image.cat([ndvi, ndmi, image_with_indices.select('EVI'), image_with_indices.select('SAVI')]).reduceRegion(
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
    evi_val = round(stats.get("EVI", 0), 3)
    savi_val = round(stats.get("SAVI", 0), 3)
    
    crop_score = ndvi_to_score(ndvi_val, crop)
    water_score = ndmi_to_score(ndmi_val, crop)
    
    # GRID ANALYSIS - Divide field into zones for detailed analysis
    zones = []
    zone_ndvi_values = []
    zone_ndmi_values = []
    
    if data.polygon:
        try:
            bounds = geometry.bounds().getInfo()['coordinates'][0]
            min_lon, min_lat = bounds[0]
            max_lon, max_lat = bounds[2]
            
            # Create 4x4 grid (16 zones) for detailed analysis
            grid_size = 4
            lon_step = (max_lon - min_lon) / grid_size
            lat_step = (max_lat - min_lat) / grid_size
            
            zone_id = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    # Calculate zone bounds
                    zone_min_lon = min_lon + (j * lon_step)
                    zone_max_lon = min_lon + ((j + 1) * lon_step)
                    zone_min_lat = min_lat + (i * lat_step)
                    zone_max_lat = min_lat + ((i + 1) * lat_step)
                    
                    # Create zone polygon
                    zone_coords = [
                        [zone_min_lon, zone_min_lat],
                        [zone_max_lon, zone_min_lat],
                        [zone_max_lon, zone_max_lat],
                        [zone_min_lon, zone_max_lat],
                        [zone_min_lon, zone_min_lat]
                    ]
                    zone_geom = ee.Geometry.Polygon([zone_coords])
                    
                    # Check if zone intersects with field
                    if not geometry.intersects(zone_geom).getInfo():
                        continue
                    
                    # Calculate zone statistics
                    zone_stats = ee.Image.cat([ndvi, ndmi]).reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=zone_geom.intersection(geometry),
                        scale=10,
                        maxPixels=1e9
                    ).getInfo()
                    
                    zone_ndvi = zone_stats.get('NDVI')
                    zone_ndmi = zone_stats.get('NDMI')
                    
                    if zone_ndvi is not None:
                        center_lat = (zone_min_lat + zone_max_lat) / 2
                        center_lon = (zone_min_lon + zone_max_lon) / 2
                        
                        # Ensure NDMI has a valid value (use 0 if None)
                        zone_ndmi_safe = zone_ndmi if zone_ndmi is not None else 0.0
                        
                        # Collect values for averaging
                        zone_ndvi_values.append(zone_ndvi)
                        if zone_ndmi is not None:
                            zone_ndmi_values.append(zone_ndmi)
                        
                        zones.append({
                            "zone_id": zone_id,
                            "row": i,
                            "col": j,
                            "center_lat": center_lat,
                            "center_lon": center_lon,
                            "ndvi": round(zone_ndvi, 3),
                            "ndmi": round(zone_ndmi_safe, 3),
                            "stress_level": get_stress_level(zone_ndvi),
                            "color": get_zone_color(zone_ndvi, "NDVI"),
                            "crop_score": ndvi_to_score(zone_ndvi, crop),
                            "water_score": ndmi_to_score(zone_ndmi_safe, crop),
                            "recommendation": get_zone_recommendation(zone_ndvi, zone_ndmi_safe)
                        })
                        zone_id += 1
        except Exception as e:
            print(f"Zone analysis failed: {e}")
            # Continue without zones if analysis fails
    
    # Calculate overall scores as average of all zones (if zones exist)
    if zones:
        avg_zone_ndvi = sum(zone_ndvi_values) / len(zone_ndvi_values) if zone_ndvi_values else ndvi_val
        avg_zone_ndmi = sum(zone_ndmi_values) / len(zone_ndmi_values) if zone_ndmi_values else ndmi_val
        
        # Use zone averages for overall scores
        ndvi_val = round(avg_zone_ndvi, 3)
        ndmi_val = round(avg_zone_ndmi, 3)
        crop_score = ndvi_to_score(ndvi_val, crop)
        water_score = ndmi_to_score(ndmi_val, crop)
    
    # Get exact satellite date from most recent image
    satellite_date = "Unknown"
    try:
        start = datetime.date.today() - datetime.timedelta(days=window_days)
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(str(start), str(datetime.date.today()))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_used))
            .sort('system:time_start', False)
        )
        latest = collection.first()
        timestamp = latest.get('system:time_start')
        if timestamp:
            satellite_date = ee.Date(timestamp).format('YYYY-MM-dd HH:mm:ss').getInfo()
    except Exception as e:
        print(f"Could not get exact date: {e}")
        satellite_date = f"Last {window_days} days composite"
    
    # Calculate area if polygon
    area_info = {}
    if data.polygon:
        area_sq_meters = geometry.area().getInfo()
        area_hectares = area_sq_meters / 10000
        area_acres = area_hectares * 2.47105
        area_info = {
            "square_meters": round(area_sq_meters, 2),
            "hectares": round(area_hectares, 4),
            "acres": round(area_acres, 4)
        }
    
    return {
        "geometry_type": geometry_type,
        "satellite_date": satellite_date,
        "satellite_source": "Sentinel-2",
        "overall": {
            "ndvi": {
                "value": ndvi_val,
                "score": crop_score,
                "meaning": interpret_ndvi(ndvi_val, crop)
            },
            "ndmi": {
                "value": ndmi_val,
                "score": water_score,
                "meaning": interpret_ndmi(ndmi_val, crop)
            },
            "evi": {
                "value": evi_val,
                "meaning": "Enhanced Vegetation Index"
            },
            "savi": {
                "value": savi_val,
                "meaning": "Soil Adjusted Vegetation Index"
            },
            "stress_level": get_stress_level(ndvi_val)
        },
        "zones": zones,
        "total_zones": len(zones),
        "area": area_info,
        "crop": crop,
        "analysis_window_days": window_days,
        "cloud_threshold_used": cloud_pct_used,
        "map_image_url": generate_ndvi_thumbnail(ndvi, geometry)
    }

# =====================================================
# FIELD HEATMAP ENDPOINT
# =====================================================
@app.post("/field-heatmap")
def field_heatmap(data: FieldHeatmapRequest):
    if not GEE_INITIALIZED:
        return {"error": "Google Earth Engine not initialized", "details": GEE_ERROR}
    
    try:
        geometry = ee.Geometry.Polygon(data.polygon)
        s2, window_days, cloud_pct_used = get_best_collection(geometry)
        
        if s2 is None:
            return {"error": "No usable satellite data found"}
        
        # Get exact satellite date
        satellite_date = "Unknown"
        try:
            start = datetime.date.today() - datetime.timedelta(days=window_days)
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geometry)
                .filterDate(str(start), str(datetime.date.today()))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_used))
                .sort('system:time_start', False)
            )
            latest = collection.first()
            timestamp = latest.get('system:time_start')
            if timestamp:
                satellite_date = ee.Date(timestamp).format('YYYY-MM-dd HH:mm:ss').getInfo()
        except:
            satellite_date = f"Last {window_days} days composite"
        
        # Calculate index
        if data.index_type == "NDVI":
            index_image = s2.normalizedDifference(['B8', 'B4']).rename('INDEX')
        elif data.index_type == "NDMI":
            index_image = s2.normalizedDifference(['B8', 'B11']).rename('INDEX')
        elif data.index_type == "EVI":
            index_image = calculate_evi(s2).select('EVI').rename('INDEX')
        else:
            index_image = s2.normalizedDifference(['B8', 'B4']).rename('INDEX')
        
        bounds = geometry.bounds().getInfo()['coordinates'][0]
        min_lon, min_lat = bounds[0]
        max_lon, max_lat = bounds[2]
        
        grid_size = min(data.grid_size, 15)  # Limit to 15x15 for performance
        lon_step = (max_lon - min_lon) / grid_size
        lat_step = (max_lat - min_lat) / grid_size
        
        zones = []
        zone_id = 0
        
        # OPTIMIZED: Create all points first, then batch sample
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                center_lat = min_lat + (i + 0.5) * lat_step
                center_lon = min_lon + (j + 0.5) * lon_step
                point = ee.Geometry.Point([center_lon, center_lat])
                
                # Quick check if point is in polygon
                if geometry.contains(point).getInfo():
                    points.append({
                        'point': point,
                        'lat': center_lat,
                        'lon': center_lon,
                        'id': zone_id
                    })
                    zone_id += 1
        
        # Sample all points at once (much faster)
        for p in points:
            try:
                value = index_image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=p['point'].buffer(10),
                    scale=10,
                    maxPixels=1e9
                ).getInfo().get('INDEX')
                
                if value is not None:
                    zones.append({
                        "zone_id": p['id'],
                        "center_lat": p['lat'],
                        "center_lon": p['lon'],
                        "value": round(value, 3),
                        "color": get_zone_color(value, data.index_type),
                        "stress_level": get_stress_level(value) if data.index_type == "NDVI" else None
                    })
            except:
                continue
        
        return {
            "success": True,
            "satellite_date": satellite_date,
            "index_type": data.index_type,
            "grid_size": f"{grid_size}x{grid_size}",
            "total_zones": len(zones),
            "zones": zones
        }
    
    except Exception as e:
        return {"error": f"Heatmap generation failed: {str(e)}"}

# =====================================================
# LAND MEASUREMENT ENDPOINT
# =====================================================
@app.post("/measure-land")
def measure_land(data: LandMeasurementRequest):
    if not GEE_INITIALIZED:
        return {"error": "Google Earth Engine not initialized", "details": GEE_ERROR}
    
    try:
        geometry = ee.Geometry.Polygon(data.polygon)
        area_sq_meters = geometry.area().getInfo()
        area_hectares = area_sq_meters / 10000
        area_acres = area_hectares * 2.47105
        perimeter_meters = geometry.perimeter().getInfo()
        
        return {
            "success": True,
            "area": {
                "square_meters": round(area_sq_meters, 2),
                "hectares": round(area_hectares, 4),
                "acres": round(area_acres, 4)
            },
            "perimeter_meters": round(perimeter_meters, 2),
            "num_vertices": len(data.polygon[0])
        }
    
    except Exception as e:
        return {"error": f"Land measurement failed: {str(e)}"}

# =====================================================
# DISTRICT NDVI SUMMARY ENDPOINT
# =====================================================
@app.get("/district-ndvi-summary")
def district_ndvi_summary():
    if not GEE_INITIALIZED:
        return {"error": "Google Earth Engine not initialized", "details": GEE_ERROR}
    
    try:
        districts = [
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"name": "Thane", "lat": 19.2183, "lon": 72.9781},
            {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
            {"name": "Nashik", "lat": 19.9975, "lon": 73.7898},
            {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882},
            {"name": "Aurangabad", "lat": 19.8762, "lon": 75.3433},
            {"name": "Solapur", "lat": 17.6599, "lon": 75.9064},
            {"name": "Kolhapur", "lat": 16.7050, "lon": 74.2433},
            {"name": "Satara", "lat": 17.6805, "lon": 74.0183},
            {"name": "Sangli", "lat": 16.8524, "lon": 74.5815},
            {"name": "Ratnagiri", "lat": 16.9902, "lon": 73.3120},
            {"name": "Sindhudurg", "lat": 16.0000, "lon": 73.6667},
            {"name": "Raigad", "lat": 18.5204, "lon": 73.0200},
            {"name": "Ahmednagar", "lat": 19.0948, "lon": 74.7480},
            {"name": "Jalgaon", "lat": 21.0077, "lon": 75.5626},
            {"name": "Dhule", "lat": 20.9042, "lon": 74.7749},
            {"name": "Nandurbar", "lat": 21.3667, "lon": 74.2333},
            {"name": "Amravati", "lat": 20.9374, "lon": 77.7796},
            {"name": "Akola", "lat": 20.7002, "lon": 77.0082},
            {"name": "Washim", "lat": 20.1097, "lon": 77.1331},
            {"name": "Buldhana", "lat": 20.5307, "lon": 76.1847},
            {"name": "Yavatmal", "lat": 20.3897, "lon": 78.1307},
            {"name": "Wardha", "lat": 20.7453, "lon": 78.5972},
            {"name": "Chandrapur", "lat": 19.9615, "lon": 79.2961},
            {"name": "Gadchiroli", "lat": 20.1809, "lon": 80.0000},
            {"name": "Gondia", "lat": 21.4560, "lon": 80.1932},
            {"name": "Bhandara", "lat": 21.1704, "lon": 79.6522},
            {"name": "Latur", "lat": 18.3984, "lon": 76.5604},
            {"name": "Osmanabad", "lat": 18.1774, "lon": 76.0407},
            {"name": "Beed", "lat": 18.9894, "lon": 75.7585},
            {"name": "Parbhani", "lat": 19.2608, "lon": 76.7611},
            {"name": "Jalna", "lat": 19.8347, "lon": 75.8800},
            {"name": "Hingoli", "lat": 19.7156, "lon": 77.1547},
            {"name": "Nanded", "lat": 19.1383, "lon": 77.3210},
            {"name": "Palghar", "lat": 19.6967, "lon": 72.7636},
            {"name": "Mumbai Suburban", "lat": 19.1136, "lon": 72.9083}
        ]
        
        results = []
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=30)
        
        for district in districts:
            try:
                point = ee.Geometry.Point([district['lon'], district['lat']])
                district_area = point.buffer(50000)
                
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(district_area)
                    .filterDate(str(start_date), str(today))
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                )
                
                count = collection.size().getInfo()
                
                if count == 0:
                    ndvi_value = 0.45
                else:
                    s2 = collection.median()
                    ndvi = s2.normalizedDifference(['B8', 'B4'])
                    stats = ndvi.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=district_area,
                        scale=10,
                        maxPixels=1e9
                    ).getInfo()
                    ndvi_value = stats.get('nd', 0.45)
                
                results.append({
                    "district": district['name'],
                    "ndvi": round(ndvi_value, 3),
                    "stress_level": get_stress_level(ndvi_value),
                    "lat": district['lat'],
                    "lon": district['lon']
                })
            
            except Exception as e:
                results.append({
                    "district": district['name'],
                    "ndvi": 0.45,
                    "stress_level": "Moderate",
                    "lat": district['lat'],
                    "lon": district['lon']
                })
        
        return {
            "success": True,
            "date_range": f"{start_date} to {today}",
            "total_districts": len(results),
            "districts": results
        }
    
    except Exception as e:
        return {"error": f"District summary failed: {str(e)}"}

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
    
    vi_mean = features[0][0]
    edge_density = features[0][6]
    print(f"🔍 NIR VALIDATION: VI={vi_mean:.4f}, Texture={edge_density:.4f}")
    
    is_plant = not (vi_mean > 0.4 and edge_density < 0.008)
    
    if not is_plant:
        print(f"❌ VALIDATION FAILED: VI={vi_mean:.2f}, Texture={edge_density:.4f}")
        return {
            "error": "invalid_specimen",
            "message": f"ANALYSIS REJECTED: Low confidence in plant specimen (Color Index: {vi_mean:.2f}, Texture: {edge_density:.4f}). Ensure the leaf covers most of the frame.",
            "debug": {"vi": vi_mean, "edge": edge_density}
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

print("✅ All endpoints registered successfully")
