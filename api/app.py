from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import os
import sys

from src.database import OracleDB

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import WineQualityModel

db: Optional[OracleDB] = None

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, db
    db = OracleDB()
    try:
        model_path = os.getenv('MODEL_PATH', 'models/wine_model.pkl')
        scaler_path = os.getenv('SCALER_PATH', 'models/scaler.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing: {model_path}")

        model = WineQualityModel(model_path=model_path, scaler_path=scaler_path)
        logger.info(f"✅ Model loaded successfully from {model_path}")
    except (FileNotFoundError, AttributeError, ImportError) as e:
        logger.error(f"❌ Critical error loading model: {e}")
        model = None
    except Exception as e:
        logger.error(f"❓ Unexpected error: {type(e).__name__}: {e}")
        model = None
    db.init_db()
    yield

    model = None
    logger.info("👋 Model unloaded")


app = FastAPI(
    title="🍷 Wine Quality Prediction API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WineFeatures(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fixed_acidity": 7.0,
                "volatile_acidity": 0.27,
                "citric_acid": 0.36,
                "residual_sugar": 20.7,
                "chlorides": 0.045,
                "free_sulfur_dioxide": 45,
                "total_sulfur_dioxide": 170,
                "density": 1.001,
                "ph": 3.0,
                "sulphates": 0.45,
                "alcohol": 8.8
            }
        },
        protected_namespaces=()
    )


    fixed_acidity: float = Field(..., description="Fixed acidity")
    volatile_acidity: float = Field(..., description="Volatile acidity")
    citric_acid: float = Field(..., description="Citric acid")
    residual_sugar: float = Field(..., description="Residual sugar")
    chlorides: float = Field(..., description="Chlorides")
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide")
    density: float = Field(..., description="Density")
    ph: float = Field(..., description="pH")
    sulphates: float = Field(..., description="Sulphates")
    alcohol: float = Field(..., description="Alcohol")


class PredictionResponse(BaseModel):
    """Модель ответа предсказания"""
    quality: str = Field(..., description="Качество вина (good/bad)")
    quality_class: int = Field(..., description="Класс качества (0/1)")
    probability: float = Field(..., description="Вероятность предсказания")
    probabilities: dict = Field(..., description="Вероятности по классам")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

    class Config:
        protected_namespaces = ()

class FeaturesResponse(BaseModel):
    """Модель ответа со списком признаков"""
    features: List[str]


class BatchPredictionRequest(BaseModel):
    """Модель для пакетного предсказания"""
    samples: List[WineFeatures]


class BatchPredictionResponse(BaseModel):
    """Модель ответа для пакетного предсказания"""
    predictions: List[PredictionResponse]


model: Optional[WineQualityModel] = None


@app.get("/", tags=["Main"])
async def root():
    """Главная страница API"""
    return {
        "message": "🍷 Wine Quality Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Проверка здоровья сервиса

    Возвращает статус сервиса и загружена ли модель
    """
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: WineFeatures):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized"
        )

    try:
        raw_dict = features.model_dump()
        features_dict = {}
        for key, value in raw_dict.items():
            new_key = key.replace('_', ' ')
            if new_key == "ph":
                new_key = "pH"
            features_dict[new_key] = value
        result = model.predict(features_dict)

        # СОХРАНЕНИЕ В БАЗУ ДАННЫХ
        if db:
            db.save_prediction(features_dict, result)
        return PredictionResponse(**result)
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        logger.error(f"Mapping error: {e}")
        raise HTTPException(status_code=422, detail=f"Missing feature in mapping: {e}")

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        results = []
        for sample in request.samples:
            features_dict = {
                'fixed acidity': sample.fixed_acidity,
                'volatile acidity': sample.volatile_acidity,
                'citric acid': sample.citric_acid,
                'residual sugar': sample.residual_sugar,
                'chlorides': sample.chlorides,
                'free sulfur dioxide': sample.free_sulfur_dioxide,
                'total sulfur dioxide': sample.total_sulfur_dioxide,
                'density': sample.density,
                'pH': sample.ph,
                'sulphates': sample.sulphates,
                'alcohol': sample.alcohol
            }
            results.append(model.predict(features_dict))

        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/features", response_model=FeaturesResponse, tags=["Info"])
async def get_features():

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return FeaturesResponse(features=model.feature_names)



@app.get("/metrics", tags=["Info"])
async def get_metrics():

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        import joblib
        metrics_path = os.getenv('METRICS_PATH', 'models/metrics.pkl')
        metrics = joblib.load(metrics_path)

        return {
            "accuracy": metrics.get('accuracy', 0),
            "roc_auc": metrics.get('roc_auc', 0),
            "train_size": metrics.get('train_size', 0),
            "test_size": metrics.get('test_size', 0),
            "feature_importance": metrics.get('feature_importance', {})
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load metrics: {str(e)}"
        )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        reload=True
    )