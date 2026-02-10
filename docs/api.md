# FastAPI NER Service - Step by Step Guide

This document explains in detail how the Bus NER API works using `src/api.py`. We'll cover the architecture, endpoints, request/response formats, and how to use the service.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Pydantic Models](#pydantic-models)
4. [Application Lifecycle](#application-lifecycle)
5. [Endpoints](#endpoints)
6. [Request/Response Examples](#requestresponse-examples)
7. [Error Handling](#error-handling)
8. [Running the Server](#running-the-server)

---

## Introduction

### What is FastAPI?

FastAPI is a modern, high-performance Python web framework for building APIs.

| Feature | Benefit |
|---------|---------|
| **Async support** | Handle many concurrent requests |
| **Auto documentation** | Swagger UI at `/docs` |
| **Type validation** | Pydantic models validate input |
| **High performance** | One of the fastest Python frameworks |

### What Does This API Do?

The API exposes the NER model as an HTTP service:

```
Client Request                    API Response
─────────────────                ──────────────────────────────
POST /ner/extract         →      {
{                                  "query": "...",
  "query": "AC bus..."             "entities": {
}                                    "SRC": "Bangalore",
                                     "DEST": "Mumbai",
                                     ...
                                   }
                                 }
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API ARCHITECTURE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────────┐
                            │     Client      │
                            │  (Browser/App)  │
                            └────────┬────────┘
                                     │ HTTP Request
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Server                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                        PYDANTIC VALIDATION                         │     │
│   │   NERRequest → validates query string                             │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                         ENDPOINT HANDLER                           │     │
│   │   @app.post("/ner/extract")                                       │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                        BusNERInference                             │     │
│   │   ner_inference.extract(query)                                    │     │
│   │   - Tokenize                                                       │     │
│   │   - Model forward pass                                            │     │
│   │   - Decode entities                                               │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                     │                                        │
│                                     ▼                                        │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │                        RESPONSE FORMATTING                         │     │
│   │   NERResponse → structured JSON output                            │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ HTTP Response
                            ┌─────────────────┐
                            │     Client      │
                            └─────────────────┘
```

---

## Pydantic Models

Pydantic models define the structure of requests and responses.

### NERRequest

**Purpose:** Validates incoming extraction requests.

```python
class NERRequest(BaseModel):
    query: str = Field(
        ...,                          # Required field
        description="Bus search query to extract entities from",
        min_length=1,                 # At least 1 character
        max_length=500,               # Max 500 characters
        examples=["AC sleeper bus from Bangalore to Mumbai tomorrow"]
    )
```

**Valid request:**
```json
{"query": "AC bus from Bangalore to Mumbai"}
```

**Invalid request (rejected):**
```json
{"query": ""}  // Too short, min_length=1
```

### EntityResult

**Purpose:** Structured entity output with all 9 entity types.

```python
class EntityResult(BaseModel):
    SRC: Optional[str] = Field(None, description="Source city")
    DEST: Optional[str] = Field(None, description="Destination city")
    BUS_TYPE: Optional[str] = Field(None, description="Bus type (AC/Non-AC)")
    SEAT_TYPE: Optional[str] = Field(None, description="Seat type (Sleeper/Seater)")
    TIME: Optional[str] = Field(None, description="Time of travel")
    DATE: Optional[str] = Field(None, description="Date of travel")
    OPERATOR: Optional[str] = Field(None, description="Bus operator")
    BOARDING_POINT: Optional[str] = Field(None, description="Pickup location")
    DROPPING_POINT: Optional[str] = Field(None, description="Drop location")
```

### NERResponse

**Purpose:** Standard response for single query extraction.

```python
class NERResponse(BaseModel):
    query: str = Field(..., description="Original query")
    entities: EntityResult = Field(..., description="Extracted entities")
```

**Example:**
```json
{
    "query": "AC bus from Bangalore to Mumbai",
    "entities": {
        "SRC": "Bangalore",
        "DEST": "Mumbai",
        "BUS_TYPE": "AC",
        "SEAT_TYPE": null,
        "TIME": null,
        "DATE": null,
        "OPERATOR": null,
        "BOARDING_POINT": null,
        "DROPPING_POINT": null
    }
}
```

### DetailedNERResponse

**Purpose:** Response with character positions for each entity.

```python
class DetailedNERResponse(BaseModel):
    query: str
    entities: EntityResult
    raw_entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of entities with positions"
    )
```

**Example:**
```json
{
    "query": "AC bus from Bangalore to Mumbai",
    "entities": {...},
    "raw_entities": [
        {"text": "AC", "label": "BUS_TYPE", "start": 0, "end": 2},
        {"text": "Bangalore", "label": "SRC", "start": 13, "end": 22},
        {"text": "Mumbai", "label": "DEST", "start": 26, "end": 32}
    ]
}
```

### BatchNERRequest / BatchNERResponse

**Purpose:** Process multiple queries in one request.

```python
class BatchNERRequest(BaseModel):
    queries: List[str] = Field(
        ...,
        description="List of queries to process",
        min_length=1,                 # At least 1 query
        max_length=100                # Max 100 queries
    )

class BatchNERResponse(BaseModel):
    results: List[NERResponse]
```

### HealthResponse

**Purpose:** Health check response.

```python
class HealthResponse(BaseModel):
    status: str               # "healthy" or "degraded"
    model_loaded: bool        # True if model is ready
    model_type: str           # "MiniLM Transformer (...)"
    model_path: str           # Path to model directory
```

---

## Application Lifecycle

### Global State

The NER model is loaded once and stored globally:

```python
# Will be initialized at startup
ner_inference: Optional[BusNERInference] = None
```

### Lifespan Manager

FastAPI's lifespan manager handles startup and shutdown:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_inference
    
    # ─────────── STARTUP ───────────
    print("STARTING BUS NER API SERVICE")
    print(f"Model path: {MODEL_PATH}")
    
    try:
        ner_inference = BusNERInference(MODEL_PATH, use_onnx=True)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("API will start but /ner/extract will fail.")
    
    yield  # ← Server runs here
    
    # ─────────── SHUTDOWN ───────────
    print("Shutting down Bus NER API...")
    ner_inference = None
```

### Why Use Lifespan?

| Approach | Issue |
|----------|-------|
| Load model per request | Slow (~seconds per request) |
| Load at import time | Blocks module import |
| **Lifespan manager** | Clean startup/shutdown, async-safe |

### Startup Flow

```
Server starts
    │
    ▼
lifespan() called
    │
    ▼
Load ONNX model (~1 second)
    │
    ▼
yield (server accepting requests)
    │
    ... server running ...
    │
Shutdown signal received
    │
    ▼
Cleanup (ner_inference = None)
    │
    ▼
Server exits
```

---

## Endpoints

### GET `/` - API Information

**Purpose:** Returns basic API info.

```python
@app.get("/", tags=["Info"])
async def root():
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
```

**Response:**
```json
{
    "name": "Bus NER API",
    "version": "2.0.0",
    "model": "MiniLM Transformer (microsoft/MiniLM-L12-H384-uncased)",
    "endpoints": {...}
}
```

### GET `/health` - Health Check

**Purpose:** Check if the service is running and model is loaded.

```python
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if ner_inference else "degraded",
        model_loaded=ner_inference is not None,
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH
    )
```

**Response (healthy):**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_type": "MiniLM Transformer (microsoft/MiniLM-L12-H384-uncased)",
    "model_path": "models/bus_ner_onnx"
}
```

**Response (degraded - model not loaded):**
```json
{
    "status": "degraded",
    "model_loaded": false,
    "model_type": "MiniLM Transformer (...)",
    "model_path": "models/bus_ner_onnx"
}
```

### POST `/ner/extract` - Single Query Extraction

**Purpose:** Extract entities from a single query.

```python
@app.post("/ner/extract", response_model=NERResponse, tags=["NER"])
async def extract_entities(request: NERRequest):
    if ner_inference is None:
        raise HTTPException(
            status_code=503,
            detail="NER model not loaded. Please train the model first."
        )
    
    try:
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
```

### POST `/ner/extract/detailed` - Detailed Extraction

**Purpose:** Extract entities with character positions.

```python
@app.post("/ner/extract/detailed", response_model=DetailedNERResponse, tags=["NER"])
async def extract_entities_detailed(request: NERRequest):
    if ner_inference is None:
        raise HTTPException(status_code=503, detail="...")
    
    try:
        result = ner_inference.extract_detailed(request.query)
        return DetailedNERResponse(
            query=result["query"],
            entities=EntityResult(**result["entities"]),
            raw_entities=result["raw_entities"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"...")
```

### POST `/ner/batch` - Batch Extraction

**Purpose:** Process multiple queries in one request.

```python
@app.post("/ner/batch", response_model=BatchNERResponse, tags=["NER"])
async def batch_extract_entities(request: BatchNERRequest):
    if ner_inference is None:
        raise HTTPException(status_code=503, detail="...")
    
    try:
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
        raise HTTPException(status_code=500, detail=f"...")
```

---

## Request/Response Examples

### Example 1: Simple Extraction

**Request:**
```bash
curl -X POST "http://localhost:8000/ner/extract" \
     -H "Content-Type: application/json" \
     -d '{"query": "AC sleeper bus from Bangalore to Mumbai tomorrow"}'
```

**Response:**
```json
{
    "query": "AC sleeper bus from Bangalore to Mumbai tomorrow",
    "entities": {
        "SRC": "Bangalore",
        "DEST": "Mumbai",
        "BUS_TYPE": "AC",
        "SEAT_TYPE": "sleeper",
        "TIME": null,
        "DATE": "tomorrow",
        "OPERATOR": null,
        "BOARDING_POINT": null,
        "DROPPING_POINT": null
    }
}
```

### Example 2: Detailed Extraction

**Request:**
```bash
curl -X POST "http://localhost:8000/ner/extract/detailed" \
     -H "Content-Type: application/json" \
     -d '{"query": "VRL bus from Chennai to Delhi"}'
```

**Response:**
```json
{
    "query": "VRL bus from Chennai to Delhi",
    "entities": {
        "SRC": "Chennai",
        "DEST": "Delhi",
        "BUS_TYPE": null,
        "SEAT_TYPE": null,
        "TIME": null,
        "DATE": null,
        "OPERATOR": "VRL",
        "BOARDING_POINT": null,
        "DROPPING_POINT": null
    },
    "raw_entities": [
        {"text": "VRL", "label": "OPERATOR", "start": 0, "end": 3},
        {"text": "Chennai", "label": "SRC", "start": 13, "end": 20},
        {"text": "Delhi", "label": "DEST", "start": 24, "end": 29}
    ]
}
```

### Example 3: Batch Extraction

**Request:**
```bash
curl -X POST "http://localhost:8000/ner/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "queries": [
         "bus from Mumbai to Pune",
         "AC sleeper from Delhi to Jaipur tomorrow"
       ]
     }'
```

**Response:**
```json
{
    "results": [
        {
            "query": "bus from Mumbai to Pune",
            "entities": {
                "SRC": "Mumbai",
                "DEST": "Pune",
                ...
            }
        },
        {
            "query": "AC sleeper from Delhi to Jaipur tomorrow",
            "entities": {
                "SRC": "Delhi",
                "DEST": "Jaipur",
                "BUS_TYPE": "AC",
                "SEAT_TYPE": "sleeper",
                "DATE": "tomorrow",
                ...
            }
        }
    ]
}
```

### Example 4: Health Check

**Request:**
```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_type": "MiniLM Transformer (microsoft/MiniLM-L12-H384-uncased)",
    "model_path": "models/bus_ner_onnx"
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| **200** | Success | Request processed successfully |
| **422** | Validation Error | Invalid request body |
| **500** | Internal Error | Model inference failed |
| **503** | Service Unavailable | Model not loaded |

### Error Response Format

```json
{
    "detail": "Error message here"
}
```

### Example: Model Not Loaded (503)

**Request:**
```bash
curl -X POST "http://localhost:8000/ner/extract" \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}'
```

**Response:**
```json
{
    "detail": "NER model not loaded. Please train the model first."
}
```

### Example: Validation Error (422)

**Request:**
```bash
curl -X POST "http://localhost:8000/ner/extract" \
     -H "Content-Type: application/json" \
     -d '{"query": ""}'
```

**Response:**
```json
{
    "detail": [
        {
            "type": "string_too_short",
            "loc": ["body", "query"],
            "msg": "String should have at least 1 character",
            "input": ""
        }
    ]
}
```

---

## Running the Server

### Development Mode

```bash
# From project root
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

- `--reload`: Auto-restart on code changes
- `--host 0.0.0.0`: Accept connections from any IP
- `--port 8000`: Listen on port 8000

### Production Mode

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

- `--workers 4`: Run 4 worker processes

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `BUS_NER_MODEL_PATH` | Override model path | `models/bus_ner_onnx` |

```bash
export BUS_NER_MODEL_PATH="/path/to/custom/model"
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Swagger Documentation

After starting the server, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation generated from your Pydantic models.

---

## Summary

The FastAPI NER service provides:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/ner/extract` | POST | Single query extraction |
| `/ner/extract/detailed` | POST | Extraction with positions |
| `/ner/batch` | POST | Batch extraction |

**Key features:**
- Model loaded once at startup (lifespan manager)
- Pydantic validation for all requests
- Structured JSON responses
- Error handling with appropriate HTTP codes
- Auto-generated Swagger documentation

The API wraps the `BusNERInference` class, exposing NER functionality over HTTP for integration with any client or service.
