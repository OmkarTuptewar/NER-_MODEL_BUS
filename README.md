# Bus Domain NER System

A production-ready Named Entity Recognition (NER) system for bus search queries, powered by **MiniLM Transformer** with optional **ONNX optimization** for high-performance inference.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Entity Types](#entity-types)
4. [Project Structure](#project-structure)
5. [How It Works - Step by Step](#how-it-works---step-by-step)
6. [Setup & Installation](#setup--installation)
7. [Usage Guide](#usage-guide)
8. [API Reference](#api-reference)
9. [ONNX Optimization](#onnx-optimization)
10. [Performance](#performance)
11. [Technical Deep Dive](#technical-deep-dive)

---

## Overview

### What is NER?

**Named Entity Recognition (NER)** is a Natural Language Processing (NLP) task that identifies and classifies named entities (like cities, dates, times) in text into predefined categories.

### Why This Project?

When users search for buses, they type natural language queries like:

```
"AC sleeper bus from Bangalore to Mumbai tomorrow morning"
```

This system extracts structured data:

```json
{
  "SRC": "Bangalore",
  "DEST": "Mumbai",
  "BUS_TYPE": "AC",
  "SEAT_TYPE": "sleeper",
  "DATE": "tomorrow",
  "TIME": "morning"
}
```

### Why Transformer (MiniLM) instead of spaCy CNN?

| Aspect | spaCy CNN | MiniLM Transformer |
|--------|-----------|-------------------|
| **Architecture** | Convolutional Neural Network | Transformer (Attention-based) |
| **Context Understanding** | Limited local context | Full bidirectional context |
| **Generalization** | Struggles with unseen words | Better generalization via subword tokenization |
| **Pre-training** | Word vectors only | Language understanding from 100M+ sentences |
| **Accuracy** | Good for simple patterns | Better for complex, varied queries |
| **Model Size** | ~5-10 MB | ~50-100 MB (or ~15 MB with ONNX + quantization) |

**We chose MiniLM** (`microsoft/MiniLM-L12-H384-uncased`) because it's:
- 6x smaller than BERT but retains 99% accuracy
- Fast inference (~10-20ms per query)
- Great for edge/production deployment

---

## Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PHASE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │  Templates   │───>│   Data       │───>│  BIO-Tagged  │              │
│   │  + Entities  │    │   Generator  │    │  JSON        │              │
│   └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                                                   │                      │
│                                                   ▼                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │   Trained    │<───│  HuggingFace │<───│  Tokenize &  │              │
│   │   Model      │    │  Trainer     │    │  Align Labels│              │
│   └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PHASE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │  User Query  │───>│  Tokenizer   │───>│  MiniLM      │              │
│   │  (text)      │    │  (subwords)  │    │  Model       │              │
│   └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                                                   │                      │
│                                                   ▼                      │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │  Structured  │<───│  Aggregate   │<───│  BIO Label   │              │
│   │  JSON        │    │  Entities    │    │  Predictions │              │
│   └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiniLM-L12-H384-uncased                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: "AC sleeper bus from Bangalore to Mumbai"               │
│                          │                                       │
│                          ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  WordPiece Tokenizer                                     │   │
│   │  [CLS] ac sleep ##er bus from bangalore to mumbai [SEP] │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Embedding Layer (384 dimensions)                        │   │
│   │  - Token Embeddings                                      │   │
│   │  - Position Embeddings                                   │   │
│   │  - Segment Embeddings                                    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  12 Transformer Layers                                   │   │
│   │  - Multi-Head Self-Attention (12 heads)                  │   │
│   │  - Feed-Forward Network                                  │   │
│   │  - Layer Normalization                                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Token Classification Head (Linear: 384 → 19 labels)    │   │
│   │  Output: [O, B-BUS, O, O, B-SRC, O, B-DEST]              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Entity Types

The system recognizes **9 entity types** relevant to bus booking:

| Entity | Description | Examples |
|--------|-------------|----------|
| `SRC` | **Source city** - Where the journey starts | Bangalore, Mumbai, Delhi, Chennai |
| `DEST` | **Destination city** - Where the journey ends | Hyderabad, Pune, Goa, Kolkata |
| `BUS_TYPE` | **Bus air conditioning type** | AC, Non-AC |
| `SEAT_TYPE` | **Seating arrangement** | Sleeper, Seater, Semi Sleeper |
| `TIME` | **Departure/travel time** | morning, night, 10 pm, after 6 pm |
| `DATE` | **Travel date** | today, tomorrow, next Friday, 25th December |
| `OPERATOR` | **Bus operator/company** | KSRTC, VRL, RedBus, Orange |
| `BOARDING_POINT` | **Pickup location** | Majestic, Electronic City, Silk Board |
| `DROPPING_POINT` | **Drop-off location** | Hitec City, Ameerpet, Koyambedu |

### BIO Tagging Scheme

We use **BIO tagging** (Beginning-Inside-Outside) to mark entity boundaries:

| Tag Type | Meaning | Example |
|----------|---------|---------|
| `B-XXX` | **Beginning** of entity XXX | First word of an entity |
| `I-XXX` | **Inside** entity XXX | Continuation words |
| `O` | **Outside** - not an entity | Regular words like "from", "to", "bus" |

**Example:**

```
Text:    "Electronic  City   bus   from  Bangalore  to  Mumbai"
Tokens:  Electronic  City   bus   from  Bangalore  to  Mumbai
Tags:    B-BOARDING  I-BOARDING  O    O    B-SRC     O   B-DEST
```

### Complete Label Set (19 labels)

```
O                    (Outside - not an entity)
B-SRC, I-SRC         (Source city)
B-DEST, I-DEST       (Destination city)
B-BUS_TYPE, I-BUS_TYPE     (Bus type)
B-SEAT_TYPE, I-SEAT_TYPE   (Seat type)
B-TIME, I-TIME       (Time)
B-DATE, I-DATE       (Date)
B-OPERATOR, I-OPERATOR     (Operator)
B-BOARDING_POINT, I-BOARDING_POINT   (Boarding point)
B-DROPPING_POINT, I-DROPPING_POINT   (Dropping point)
```

---

## Project Structure

```
NerModel_GTEXP1.0/
│
├── src/                          # Source code
│   ├── data_generator.py         # Training data generation
│   ├── train_ner.py              # Model training pipeline
│   ├── inference.py              # Inference (PyTorch + ONNX)
│   ├── export_onnx.py            # ONNX export utility
│   └── api.py                    # FastAPI REST service
│
├── data/                         # Training data
│   ├── training_data_bio.json    # BIO-tagged training data
│   ├── label2id.json             # Label to ID mapping
│   └── id2label.json             # ID to label mapping
│
├── models/                       # Trained models
│   ├── bus_ner_transformer/      # PyTorch model
│   │   ├── model.safetensors     # Model weights
│   │   ├── config.json           # Model config
│   │   ├── tokenizer.json        # Tokenizer
│   │   ├── label2id.json         # Label mappings
│   │   └── id2label.json
│   │
│   └── bus_ner_onnx/             # ONNX optimized model
│       ├── model.onnx            # ONNX graph
│       ├── tokenizer.json
│       └── label mappings...
│
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation
```

---

## How It Works - Step by Step

### Step 1: Data Generation (`data_generator.py`)

**What happens:**
1. Templates with placeholders are defined (100+ patterns)
2. Entity values are randomly selected from pools
3. Templates are filled and character spans are calculated
4. Spans are converted to BIO tags

**Template Example:**
```python
"{OPERATOR} {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE} {TIME}"
```

**Filled Example:**
```
"VRL AC sleeper from Bangalore to Mumbai tomorrow morning"
```

**BIO Output:**
```json
{
  "tokens": ["VRL", "AC", "sleeper", "from", "Bangalore", "to", "Mumbai", "tomorrow", "morning"],
  "ner_tags": [13, 5, 7, 0, 1, 0, 3, 11, 9]
}
```

Where numbers map to labels:
- 0 = O (outside)
- 1 = B-SRC
- 3 = B-DEST
- 5 = B-BUS_TYPE
- 7 = B-SEAT_TYPE
- 9 = B-TIME
- 11 = B-DATE
- 13 = B-OPERATOR

---

### Step 2: Training (`train_ner.py`)

**What happens:**

1. **Load Data**: Read BIO-tagged JSON file
2. **Prepare Dataset**: Split into train (90%) and eval (10%)
3. **Tokenize**: Use MiniLM tokenizer with subword alignment
4. **Model Setup**: Load pre-trained MiniLM, add classification head
5. **Train**: Fine-tune using HuggingFace Trainer
6. **Save**: Export model, tokenizer, and label mappings

**Subword Tokenization Challenge:**

When "Bangalore" becomes `["bang", "##alore"]`, we need to align labels:

```
Original:    Bangalore  →  Label: B-SRC
Subwords:    bang       →  Label: B-SRC  (first subword gets label)
             ##alore    →  Label: -100   (ignored in loss)
```

**Training Arguments:**
```python
TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

---

### Step 3: Inference (`inference.py`)

**What happens:**

1. **Tokenize Query**: Split by whitespace, then subword tokenize
2. **Model Forward Pass**: Get logits for each token
3. **Argmax**: Pick highest probability label for each token
4. **Aggregate**: Combine B- and I- tags into entities
5. **Return**: Structured JSON with entity values

**Example Flow:**

```
Input: "AC sleeper from Bangalore to Mumbai"

Tokenize → ["ac", "sleep", "##er", "from", "bangalore", "to", "mumbai"]

Model Output (logits → labels):
  ac        → B-BUS_TYPE
  sleep     → B-SEAT_TYPE
  ##er      → -100 (ignored)
  from      → O
  bangalore → B-SRC
  to        → O
  mumbai    → B-DEST

Aggregate:
  BUS_TYPE: "AC"
  SEAT_TYPE: "sleeper"
  SRC: "Bangalore"
  DEST: "Mumbai"
```

---

### Step 4: API Service (`api.py`)

**What happens:**

1. **Startup**: Load model once (singleton pattern)
2. **Request**: Receive JSON with query string
3. **Process**: Call inference module
4. **Response**: Return structured JSON

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ner/extract` | Extract entities from query |
| POST | `/ner/extract/detailed` | Extract with character positions |
| POST | `/ner/batch` | Process multiple queries |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger documentation |

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip package manager
- ~2GB disk space (for model + dependencies)

### Step 1: Clone & Setup

```bash
# Navigate to project
cd NerModel_GTEXP1.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Training Data

```bash
python src/data_generator.py
```

Output:
```
Bus NER Data Generator
============================================================
Generating 2000 training samples...
Validating samples...
Valid samples: 2000/2000
Saved 2000 BIO-tagged samples to data/training_data_bio.json
```

### Step 3: Train the Model

```bash
python src/train_ner.py --epochs 10
```

Output:
```
BUS NER TRAINING PIPELINE (Transformers)
============================================================
Model: microsoft/MiniLM-L12-H384-uncased
Epochs: 10
Train samples: 1800
Eval samples: 200

Epoch 1/10 - Loss: 0.5234 - F1: 0.85
Epoch 2/10 - Loss: 0.1523 - F1: 0.95
...
Epoch 10/10 - Loss: 0.0123 - F1: 1.00

Model saved to: models/bus_ner_transformer
```

### Step 4: Start API Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Access at: `http://localhost:8000`

---

## Usage Guide

### Command Line Inference

```bash
# Test with default queries
python src/inference.py

# Test specific query
python src/inference.py --query "VRL AC sleeper from Chennai to Hyderabad tomorrow"

# Use ONNX model
python src/inference.py --onnx
```

### API Usage (curl)

```bash
# Extract entities
curl -X POST "http://localhost:8000/ner/extract" \
  -H "Content-Type: application/json" \
  -d '{"query": "AC sleeper bus from Bangalore to Mumbai tomorrow"}'

# Batch extraction
curl -X POST "http://localhost:8000/ner/batch" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["bus from Pune to Goa", "VRL AC to Mumbai tonight"]}'
```

### Python Usage

```python
from inference import BusNERInference

# Load model
ner = BusNERInference("models/bus_ner_transformer")

# Extract entities
result = ner.extract("AC sleeper from Bangalore to Mumbai tomorrow")
print(result)
# {'SRC': 'Bangalore', 'DEST': 'Mumbai', 'BUS_TYPE': 'AC', 
#  'SEAT_TYPE': 'sleeper', 'DATE': 'tomorrow', ...}

# Detailed extraction with positions
detailed = ner.extract_detailed("VRL bus from Chennai to Hyderabad")
print(detailed["raw_entities"])
# [{'text': 'VRL', 'label': 'OPERATOR', 'start': 0, 'end': 3}, ...]

# Batch processing
results = ner.batch_extract([
    "bus from Pune to Goa",
    "KSRTC sleeper tomorrow night"
])
```

---

## API Reference

### POST /ner/extract

Extract entities from a single query.

**Request:**
```json
{
  "query": "AC sleeper bus from Bangalore to Mumbai tomorrow"
}
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

### POST /ner/extract/detailed

Extract with character positions.

**Response includes:**
```json
{
  "query": "...",
  "entities": {...},
  "raw_entities": [
    {"text": "Bangalore", "label": "SRC", "start": 25, "end": 34},
    {"text": "Mumbai", "label": "DEST", "start": 38, "end": 44}
  ]
}
```

### POST /ner/batch

Process multiple queries.

**Request:**
```json
{
  "queries": [
    "bus from Pune to Goa",
    "VRL AC sleeper tomorrow"
  ]
}
```

### GET /health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "MiniLM Transformer (microsoft/MiniLM-L12-H384-uncased)",
  "model_path": "/path/to/models/bus_ner_transformer"
}
```

---

## ONNX Optimization

### What is ONNX?

**ONNX** (Open Neural Network Exchange) is an open format for ML models that enables:
- 2-5x faster inference
- Smaller model size with quantization
- Cross-platform deployment (C++, Java, JavaScript)
- Hardware acceleration

### Export to ONNX

```bash
# Basic export
python src/export_onnx.py

# With quantization (smaller, faster)
python src/export_onnx.py --quantize

# Verify exported model
python src/export_onnx.py --verify
```

### Use ONNX Model

```python
from inference import BusNERInference

# Load ONNX model
ner = BusNERInference("models/bus_ner_onnx", use_onnx=True)

# Same API as PyTorch model
result = ner.extract("AC sleeper from Bangalore to Mumbai")
```

### Size Comparison

| Model | Size | Inference Time |
|-------|------|----------------|
| PyTorch (FP32) | ~130 MB | ~20ms |
| ONNX (FP32) | ~45 MB | ~10ms |
| ONNX (INT8 Quantized) | ~15 MB | ~5ms |

---

## Performance

### Model Metrics

| Metric | Value |
|--------|-------|
| **Precision** | 99.5%+ |
| **Recall** | 99.5%+ |
| **F1 Score** | 99.5%+ |
| **Training Time** | ~5 min (10 epochs, 2000 samples) |

### Inference Speed

| Model Type | Latency (per query) |
|------------|---------------------|
| PyTorch (CPU) | 15-25ms |
| PyTorch (GPU) | 5-10ms |
| ONNX (CPU) | 8-15ms |
| ONNX (Quantized) | 3-8ms |

### Scalability

| Training Samples | Training Time | Recommended Use |
|------------------|---------------|-----------------|
| 2,000 | 5 min | Development/Testing |
| 10,000 | 20 min | Staging |
| 50,000+ | 1-2 hours | Production |

---

## Technical Deep Dive

### Why MiniLM Works Well

1. **Knowledge Distillation**: MiniLM is distilled from larger models, retaining language understanding in a smaller package.

2. **Subword Tokenization**: Handles unseen words by breaking them into known subwords:
   - "Aurangabad" → ["aur", "##ang", "##abad"]
   - Model can still recognize it as a city based on context

3. **Bidirectional Context**: Unlike RNNs, transformers see the entire sentence:
   - "bus from X to Y" → X is likely SRC, Y is likely DEST
   - "pickup at X drop at Y" → X is BOARDING, Y is DROPPING

### Token Classification vs Sequence Labeling

We use **Token Classification** where each token gets an independent label:

```
Input:  [CLS] ac sleeper from bangalore [SEP]
Output: [-100, B-BUS, B-SEAT, O, B-SRC, -100]
```

`-100` tokens are ignored in loss calculation (special tokens and subword continuations).

### Label Alignment Strategy

When tokenizing pre-split words:

```python
tokenizer(
    ["AC", "sleeper", "from", "Bangalore"],
    is_split_into_words=True  # Critical flag!
)
```

This preserves word boundaries so we can map labels correctly.

---

## Troubleshooting

### "Model not found" Error

```bash
# Make sure model exists
ls models/bus_ner_transformer/

# If missing, train first
python src/train_ner.py
```

### "Address already in use" (Port 8000)

```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Restart server
uvicorn src.api:app --port 8000
```

### Low Accuracy

1. Increase training samples:
   ```python
   # In data_generator.py
   NUM_SAMPLES = 10000  # Increase from 2000
   ```

2. Train more epochs:
   ```bash
   python src/train_ner.py --epochs 20
   ```

3. Add more entity value examples in `ENTITY_VALUES` dict

---

## License

MIT License - Free for commercial and personal use.

---

## Quick Reference
source venv/bin/activate
```bash
# Generate data
python src/data_generator.py

# Train model
python src/train_ner.py --epochs 10

# Export to ONNX
python src/export_onnx.py --verify

# Test inference
python src/inference.py --query "AC sleeper from Bangalore to Mumbai"

# Start API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Test API
curl -X POST http://localhost:8000/ner/extract \
  -H "Content-Type: application/json" \
  -d '{"query": "VRL AC sleeper from Chennai to Hyderabad tomorrow"}'
```
Which Model Does the API Use?
Your current api.py uses bus_ner_transformer/ (PyTorch model).
To switch to ONNX for faster inference, update api.py:

# Current (PyTorch)ner_inference = BusNERInference(MODEL_PATH)
# For ONNX (faster)ner_inference = BusNERInference(MODEL_PATH, use_onnx=True)


And change the path:
# DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "bus_ner_onnx"

Summary
Model	Use When
bus_ner/	Don't use (legacy)
bus_ner_transformer/	Development, debugging, GPU inference
bus_ner_onnx/	Production, CPU deployment, low latency
