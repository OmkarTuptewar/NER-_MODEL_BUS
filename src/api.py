

import os
import sys
from pathlib import Path
from typing import Optional,Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from inference import BusNERInference

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model info
MODEL_TYPE = os.environ.get(
    "BUS_NER_MODEL_TYPE",
    "MiniLM Transformer (microsoft/MiniLM-L12-H384-uncased)",
)

USE_ONNX = os.environ.get("BUS_NER_USE_ONNX", "true").lower() in {"1", "true", "yes"}

# Model path - can be overridden by environment variables
DEFAULT_TORCH_PATH = Path(__file__).parent.parent / "models" / "bus_ner_transformer_v5"
DEFAULT_ONNX_PATH = Path(__file__).parent.parent / "models" / "bus_ner_onnx_v5"

if USE_ONNX:
    MODEL_PATH = os.environ.get("BUS_NER_ONNX_MODEL_PATH", str(DEFAULT_ONNX_PATH))
else:
    MODEL_PATH = os.environ.get("BUS_NER_MODEL_PATH", str(DEFAULT_TORCH_PATH))

def _resolve_env_path(path_value: str) -> str:
    p = Path(path_value)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p
    return str(p)

MODEL_PATH = _resolve_env_path(MODEL_PATH)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class NERRequest(BaseModel):
    """Request model for NER extraction."""
    query: str = Field(
        ...,
        description="Bus search query to extract entities from",
        min_length=1,
        max_length=1000,
        examples=["AC sleeper bus from Bangalore to Mumbai tomorrow"]
 )


class EntityResult(BaseModel):
    """Structured entity extraction result."""
    SOURCE_NAME: List[str] = Field(default_factory=list)
    SOURCE_CITY_CODE: List[str] = Field(default_factory=list)
    DESTINATION_NAME: List[str] = Field(default_factory=list)
    DESTINATION_CITY_CODE: List[str] = Field(default_factory=list)
    DEPARTURE_DATE: List[str] = Field(default_factory=list)
    ARRIVAL_DATE: List[str] = Field(default_factory=list)
    DEPARTURE_TIME: List[str] = Field(default_factory=list)
    ARRIVAL_TIME: List[str] = Field(default_factory=list)
    PICKUP_POINT: List[str] = Field(default_factory=list)
    DROP_POINT: List[str] = Field(default_factory=list)
    AC_TYPE: List[str] = Field(default_factory=list)
    BUS_TYPE: List[str] = Field(default_factory=list)
    SEAT_TYPE: List[str] = Field(default_factory=list)
    AMENITIES: List[str] = Field(default_factory=list)
    BUS_FEATURES: List[str] = Field(default_factory=list)
    OPERATOR: List[str] = Field(default_factory=list)
    COUPON_CODE: List[str] = Field(default_factory=list)
    DEALS: List[str] = Field(default_factory=list)
    ADD_ONS: List[str] = Field(default_factory=list)
    PRICE: List[str] = Field(default_factory=list)
    SEMANTIC: List[str] = Field(default_factory=list)
    TRAVELER: List[str] = Field(default_factory=list)

class NERResponse(BaseModel):
    """Response model for NER extraction."""
    query: str = Field(..., description="Original query")
    entities: EntityResult = Field(..., description="Extracted entities")


class DetailedNERResponse(BaseModel):
    """Detailed response with raw entity positions."""
    query: str
    entities: EntityResult
    raw_entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of entities with positions"
    )


class BatchNERRequest(BaseModel):
    """Request model for batch NER extraction."""
    queries: List[str] = Field(
        ...,
        description="List of queries to process",
        min_length=1,
        max_length=100
    )


class BatchNERResponse(BaseModel):
    """Response model for batch NER extraction."""
    results: List[NERResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: str
    model_path: str


# =============================================================================
# GLOBAL STATE
# =============================================================================

# Will be initialized at startup
ner_inference: Optional[BusNERInference] = None


# =============================================================================
# LIFESPAN MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan.
    
    Loads the NER transformer model at startup and handles cleanup at shutdown.
    """
    global ner_inference
    
    print("=" * 60)
    print("STARTING BUS NER API SERVICE (Transformer)")
    print("=" * 60)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Use ONNX: {USE_ONNX}")
    
    try:
        ner_inference = BusNERInference(MODEL_PATH, use_onnx=USE_ONNX)
        print("Model loaded successfully!")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("API will start but /ner/extract will fail.")
        print("Please train the model first using: python src/train_ner.py")
        print("=" * 60)
    except Exception as e:
        print(f"WARNING: Failed to load model: {e}")
        print("API will start but /ner/extract will fail.")
        print("=" * 60)
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down Bus NER API...")
    ner_inference = None


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Bus NER API",
    description="""
    Named Entity Recognition API for bus search queries.
    
    Powered by MiniLM Transformer (microsoft/MiniLM-L12-H384-uncased).
    
    Extracts structured entities from natural language bus queries including:
    - Source and destination cities
    - Bus type (AC/Non-AC)
    - Seat type (Sleeper/Seater)
    - Travel time and date
    - Bus operator
    - Boarding and dropping points
    """,
    version="2.0.0",
    lifespan=lifespan
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Bus NER API",
        "version": "2.0.0",
        "model": MODEL_TYPE,
        "endpoints": {
            "extract": "POST /ner/extract",
            "extract_detailed": "POST /ner/extract/detailed",
            "batch": "POST /ner/batch",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy" if ner_inference else "degraded",
        model_loaded=ner_inference is not None,
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH
    )


@app.post("/ner/extract", response_model=NERResponse, tags=["NER"])
async def extract_entities(request: NERRequest):
    """
    Extract entities from a bus search query.
    
    Takes a natural language bus query and returns structured entities.
    
    Example:
        Input: {"query": "AC sleeper bus from Bangalore to Mumbai tomorrow"}
        Output: {
            "query": "AC sleeper bus from Bangalore to Mumbai tomorrow",
            "entities": {
                "SRC": "Bangalore",
                "DEST": "Mumbai",
                "BUS_TYPE": "AC",
                "SEAT_TYPE": "sleeper",
                "DATE": "tomorrow",
                ...
            }
        }
    """
    if ner_inference is None:
        raise HTTPException(
            status_code=503,
            detail="NER model not loaded. Please train the model first."
        )
    
    try:
        # Extract entities
        entities = ner_inference.extract(request.query)
        
        return NERResponse(
            query=request.query,
            entities=EntityResult(**entities)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during entity extraction: {str(e)}"
        )


@app.post("/ner/extract/detailed", response_model=DetailedNERResponse, tags=["NER"])
async def extract_entities_detailed(request: NERRequest):
    """
    Extract entities with detailed position information.
    
    Returns both structured entities and raw entity list with character positions.
    """
    if ner_inference is None:
        raise HTTPException(
            status_code=503,
            detail="NER model not loaded. Please train the model first."
        )
    
    try:
        result = ner_inference.extract_detailed(request.query)
        
        return DetailedNERResponse(
            query=result["query"],
            entities=EntityResult(**result["entities"]),
            raw_entities=result["raw_entities"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during entity extraction: {str(e)}"
        )


@app.post("/ner/batch", response_model=BatchNERResponse, tags=["NER"])
async def batch_extract_entities(request: BatchNERRequest):
    """
    Extract entities from multiple queries in a single request.
    
    More efficient than making individual requests for each query.
    """
    if ner_inference is None:
        raise HTTPException(
            status_code=503,
            detail="NER model not loaded. Please train the model first."
        )
    
    try:
        # Use batch extraction
        results = ner_inference.batch_extract(request.queries)
        
        responses = [
            NERResponse(
                query=query,
                entities=EntityResult(**entities)
            )
            for query, entities in zip(request.queries, results)
        ]
        
        return BatchNERResponse(results=responses)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during batch extraction: {str(e)}"
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Bus NER API server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
