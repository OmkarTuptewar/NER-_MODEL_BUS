# Inference Process - Step by Step Guide

This document explains in detail how the Bus NER model performs inference using `src/inference.py`. We'll walk through the complete flow from receiving a query to returning extracted entities.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Inference Flow Overview](#inference-flow-overview)
3. [BusNERInference Class](#busneinference-class)
4. [Model Loading Functions](#model-loading-functions)
5. [The Prediction Pipeline](#the-prediction-pipeline)
6. [Entity Extraction Functions](#entity-extraction-functions)
7. [Step-by-Step Example](#step-by-step-example)
8. [PyTorch vs ONNX Inference](#pytorch-vs-onnx-inference)

---

## Introduction

### What is Inference?

**Inference** is the process of using a trained model to make predictions on new data. In our case:

| Phase | What Happens |
|-------|--------------|
| **Training** | Model learns patterns from labeled data |
| **Inference** | Model applies learned patterns to new queries |

### Two Inference Modes

The inference module supports two backends:

| Mode | Library | Speed | Use Case |
|------|---------|-------|----------|
| **PyTorch** | transformers | Baseline | Development, GPU inference |
| **ONNX** | optimum + onnxruntime | 2-5x faster | Production, CPU deployment |

```python
# PyTorch mode (default)
ner = BusNERInference("models/bus_ner_transformer")

# ONNX mode (faster)
ner = BusNERInference("models/bus_ner_onnx", use_onnx=True)
```

---

## Inference Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  User Query: "AC sleeper bus from Bangalore to Mumbai tomorrow"
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 1: WHITESPACE TOKENIZATION                            │
  │  words = query.split()                                      │
  │  → ["AC", "sleeper", "bus", "from", "Bangalore", "to",      │
  │     "Mumbai", "tomorrow"]                                   │
  └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 2: SUBWORD TOKENIZATION                               │
  │  tokenizer(words, is_split_into_words=True)                 │
  │  → [CLS] ac sleep ##er bus from bangalore to mumbai         │
  │     tomorrow [SEP]                                          │
  └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 3: MODEL FORWARD PASS                                 │
  │  outputs = model(**inputs)                                  │
  │  predictions = argmax(outputs.logits)                       │
  │  → [-, 5, 7, -, 0, 0, 1, 0, 3, 11, -]                       │
  └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 4: MAP PREDICTIONS TO WORDS                           │
  │  Only keep first subword of each word                       │
  │  → AC=5, sleeper=7, bus=0, from=0, Bangalore=1,             │
  │     to=0, Mumbai=3, tomorrow=11                             │
  └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 5: DECODE BIO TAGS TO ENTITIES                        │
  │  Collect B-X and following I-X tokens                       │
  │  → [("AC", "BUS_TYPE"), ("sleeper", "SEAT_TYPE"),           │
  │     ("Bangalore", "SRC"), ("Mumbai", "DEST"),               │
  │     ("tomorrow", "DATE")]                                   │
  └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 6: FORMAT OUTPUT                                      │
  │  {                                                          │
  │    "SRC": "Bangalore",                                      │
  │    "DEST": "Mumbai",                                        │
  │    "BUS_TYPE": "AC",                                        │
  │    "SEAT_TYPE": "sleeper",                                  │
  │    "DATE": "tomorrow",                                      │
  │    "TIME": null, "OPERATOR": null, ...                      │
  │  }                                                          │
  └─────────────────────────────────────────────────────────────┘
```

---

## BusNERInference Class

### Class Overview

```python
class BusNERInference:
    """
    Inference wrapper for the Bus NER model.
    Supports both PyTorch and ONNX models.
    """
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_path` | Path | Directory containing model files |
| `use_onnx` | bool | True for ONNX, False for PyTorch |
| `tokenizer` | AutoTokenizer | HuggingFace tokenizer |
| `model` | Model | PyTorch or ONNX model |
| `id2label` | Dict[int, str] | Maps IDs to labels (0 → "O") |
| `label2id` | Dict[str, int] | Maps labels to IDs ("O" → 0) |
| `entity_types` | List[str] | All 9 entity type names |
| `device` | str | "cpu" or "cuda" |

### Constructor: `__init__(model_path, use_onnx=False)`

```python
def __init__(self, model_path: str, use_onnx: bool = False):
    self.model_path = Path(model_path)
    self.use_onnx = use_onnx
    self.entity_types = ALL_ENTITY_TYPES
    
    if use_onnx:
        self.tokenizer, self.model, self.id2label, self.label2id = self._load_onnx_model()
    else:
        self.tokenizer, self.model, self.id2label, self.label2id = self._load_pytorch_model()
```

**What happens:**
1. Store model path and mode
2. Set entity types (9 types: SRC, DEST, etc.)
3. Load appropriate model based on `use_onnx` flag

---

## Model Loading Functions

### Function: `_load_pytorch_model()`

**What it does:** Loads the PyTorch transformer model from disk.

**Steps:**
1. Check if model directory exists
2. Set device (GPU if available, else CPU)
3. Load tokenizer
4. Load model and move to device
5. Set model to evaluation mode
6. Load label mappings

```python
def _load_pytorch_model(self):
    # Set device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(self.model_path)
    model.to(self.device)
    model.eval()  # Important! Disables dropout
    
    # Load label mappings
    label2id, id2label = self._load_label_mappings()
    
    return tokenizer, model, id2label, label2id
```

### Function: `_load_onnx_model()`

**What it does:** Loads the ONNX-optimized model.

**Differences from PyTorch:**
- Uses `ORTModelForTokenClassification` from optimum library
- Always runs on CPU (ONNX Runtime)
- Faster inference, no GPU required

```python
def _load_onnx_model(self):
    from optimum.onnxruntime import ORTModelForTokenClassification
    
    self.device = "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    model = ORTModelForTokenClassification.from_pretrained(self.model_path)
    
    label2id, id2label = self._load_label_mappings()
    
    return tokenizer, model, id2label, label2id
```

### Function: `_load_label_mappings()`

**What it does:** Loads the label-to-ID dictionaries from JSON files.

```python
def _load_label_mappings(self):
    # Load label2id.json
    with open(self.model_path / "label2id.json", 'r') as f:
        label2id = json.load(f)
    
    # Load id2label.json
    with open(self.model_path / "id2label.json", 'r') as f:
        id2label_str = json.load(f)
        id2label = {int(k): v for k, v in id2label_str.items()}
    
    return label2id, id2label
```

**Example mappings:**
```python
label2id = {"O": 0, "B-SRC": 1, "I-SRC": 2, "B-DEST": 3, ...}
id2label = {0: "O", 1: "B-SRC", 2: "I-SRC", 3: "B-DEST", ...}
```

---

## The Prediction Pipeline

### Function: `_predict(text)`

This is the core function that runs the model and extracts entities.

**Input:** Query string  
**Output:** List of `(entity_text, label, start_char, end_char)` tuples

### Step 1: Whitespace Tokenization

```python
words = text.split()
# "AC bus from Bangalore" → ["AC", "bus", "from", "Bangalore"]
```

### Step 2: Subword Tokenization

```python
encoding = self.tokenizer(
    words,
    is_split_into_words=True,  # Input is already tokenized by words
    return_tensors="pt",
    truncation=True,
    padding=True,
)
```

**Result:**
```
words:     ["AC",  "bus", "from", "Bangalore"]
subwords:  [CLS]   ac     bus    from   ban    ##gal  ##ore  [SEP]
word_ids:  None    0      1      2      3      3      3      None
```

### Step 3: Model Forward Pass

**PyTorch:**
```python
with torch.no_grad():
    outputs = self.model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
pred_ids = predictions[0].cpu().numpy()
```

**ONNX:**
```python
outputs = self.model(**encoding)
logits = outputs.logits
predictions = np.argmax(logits.numpy(), axis=2)
pred_ids = predictions[0]
```

### Step 4: Map Predictions to Words

Only keep the prediction for the **first subword** of each word:

```python
word_ids = encoding.word_ids()
word_labels = []
prev_word_idx = None

for idx, word_idx in enumerate(word_ids):
    if word_idx is not None and word_idx != prev_word_idx:
        label = self.id2label[pred_ids[idx]]
        word_labels.append((word_idx, label))
    prev_word_idx = word_idx
```

**Result:**
```python
word_labels = [
    (0, "B-BUS_TYPE"),   # AC
    (1, "O"),            # bus
    (2, "O"),            # from
    (3, "B-SRC"),        # Bangalore
]
```

### Step 5: Decode BIO Tags to Entities

The state machine that collects consecutive B/I tokens:

```python
entities = []
current_entity_words = []
current_entity_label = None

for word_idx, label in word_labels:
    if label.startswith("B-"):
        # Save previous entity if exists
        if current_entity_words:
            entities.append((entity_text, current_entity_label, ...))
        
        # Start new entity
        current_entity_words = [words[word_idx]]
        current_entity_label = label[2:]  # Remove "B-" prefix
        
    elif label.startswith("I-") and current_entity_label == label[2:]:
        # Continue current entity
        current_entity_words.append(words[word_idx])
        
    else:  # "O" or mismatched I-
        # End current entity
        if current_entity_words:
            entities.append((entity_text, current_entity_label, ...))
        current_entity_words = []
```

### Function: `_get_char_positions(text, words, start_word_idx, end_word_idx)`

**What it does:** Calculates character positions for an entity span.

```python
def _get_char_positions(self, text, words, start_word_idx, end_word_idx):
    # Calculate start position
    start_char = 0
    for i in range(start_word_idx):
        start_char += len(words[i]) + 1  # +1 for space
    
    # Calculate end position
    end_char = start_char
    for i in range(start_word_idx, end_word_idx + 1):
        end_char += len(words[i])
        if i < end_word_idx:
            end_char += 1  # Space between words
    
    return start_char, end_char
```

**Example:**
```
text = "bus from Bangalore to Mumbai"
words = ["bus", "from", "Bangalore", "to", "Mumbai"]

For "Bangalore" (word index 2):
  start_char = len("bus") + 1 + len("from") + 1 = 9
  end_char = 9 + len("Bangalore") = 18
```

---

## Entity Extraction Functions

### Function: `extract(query)`

**What it does:** Returns a simple dictionary with entity values.

```python
def extract(self, query: str) -> Dict[str, Optional[str]]:
    entities = self._predict(query)
    
    # Initialize all entity types as None
    result = {entity_type: None for entity_type in self.entity_types}
    
    # Fill in detected entities (first occurrence only)
    for entity_text, label, _, _ in entities:
        if label in result and result[label] is None:
            result[label] = entity_text
    
    return result
```

**Example output:**
```python
{
    "SRC": "Bangalore",
    "DEST": "Mumbai",
    "BUS_TYPE": "AC",
    "SEAT_TYPE": "sleeper",
    "TIME": None,
    "DATE": "tomorrow",
    "OPERATOR": None,
    "BOARDING_POINT": None,
    "DROPPING_POINT": None
}
```

### Function: `extract_detailed(query)`

**What it does:** Returns entities with character positions.

```python
def extract_detailed(self, query: str) -> Dict[str, Any]:
    entity_tuples = self._predict(query)
    
    entities = {entity_type: None for entity_type in self.entity_types}
    raw_entities = []
    
    for entity_text, label, start_char, end_char in entity_tuples:
        raw_entities.append({
            "text": entity_text,
            "label": label,
            "start": start_char,
            "end": end_char
        })
        
        if label in entities and entities[label] is None:
            entities[label] = entity_text
    
    return {
        "query": query,
        "entities": entities,
        "raw_entities": raw_entities
    }
```

**Example output:**
```python
{
    "query": "AC bus from Bangalore to Mumbai",
    "entities": {
        "SRC": "Bangalore",
        "DEST": "Mumbai",
        "BUS_TYPE": "AC",
        ...
    },
    "raw_entities": [
        {"text": "AC", "label": "BUS_TYPE", "start": 0, "end": 2},
        {"text": "Bangalore", "label": "SRC", "start": 13, "end": 22},
        {"text": "Mumbai", "label": "DEST", "start": 26, "end": 32}
    ]
}
```

### Function: `batch_extract(queries)`

**What it does:** Processes multiple queries.

```python
def batch_extract(self, queries: List[str]) -> List[Dict[str, Optional[str]]]:
    results = []
    for query in queries:
        result = self.extract(query)
        results.append(result)
    return results
```

**Note:** Currently processes queries one at a time. For true batching, the model forward pass would need modification.

---

## Step-by-Step Example

Let's trace through a complete example:

### Input

```python
query = "VRL AC sleeper from Chennai to Delhi tomorrow"
```

### Step 1: Whitespace Split

```python
words = ["VRL", "AC", "sleeper", "from", "Chennai", "to", "Delhi", "tomorrow"]
```

### Step 2: Subword Tokenization

```
Position:   0      1     2    3       4      5       6     7      8      9      10
Subword:  [CLS]  vrl   ac  sleep  ##er   from  chennai  to   delhi  tomorrow [SEP]
Word_id:   None   0     1    2      2      3      4      5     6       7     None
```

### Step 3: Model Predictions

```
Position:   0      1     2    3       4      5      6     7     8       9      10
Subword:  [CLS]  vrl   ac  sleep  ##er   from  chennai  to   delhi  tomorrow [SEP]
Pred_id:    -     13    5    7      -      0      1      0     3      11      -
```

### Step 4: Map to Words (First Subword Only)

```python
word_labels = [
    (0, "B-OPERATOR"),    # VRL → 13
    (1, "B-BUS_TYPE"),    # AC → 5
    (2, "B-SEAT_TYPE"),   # sleeper → 7
    (3, "O"),             # from → 0
    (4, "B-SRC"),         # Chennai → 1
    (5, "O"),             # to → 0
    (6, "B-DEST"),        # Delhi → 3
    (7, "B-DATE"),        # tomorrow → 11
]
```

### Step 5: Decode to Entities

```python
entities = [
    ("VRL", "OPERATOR", 0, 3),
    ("AC", "BUS_TYPE", 4, 6),
    ("sleeper", "SEAT_TYPE", 7, 14),
    ("Chennai", "SRC", 20, 27),
    ("Delhi", "DEST", 31, 36),
    ("tomorrow", "DATE", 37, 45)
]
```

### Step 6: Final Output

```python
{
    "SRC": "Chennai",
    "DEST": "Delhi",
    "BUS_TYPE": "AC",
    "SEAT_TYPE": "sleeper",
    "TIME": None,
    "DATE": "tomorrow",
    "OPERATOR": "VRL",
    "BOARDING_POINT": None,
    "DROPPING_POINT": None
}
```

---

## PyTorch vs ONNX Inference

### Performance Comparison

| Aspect | PyTorch | ONNX |
|--------|---------|------|
| **Latency** | ~20-40ms | ~5-15ms |
| **CPU Usage** | Higher | Optimized |
| **GPU Support** | Yes | Optional |
| **Model Size** | ~50 MB | ~50 MB (or ~15 MB quantized) |
| **Dependencies** | transformers, torch | optimum, onnxruntime |

### When to Use Which

| Scenario | Recommended |
|----------|-------------|
| Development/debugging | PyTorch |
| GPU available | PyTorch |
| Production CPU deployment | ONNX |
| Low latency required | ONNX |
| Edge devices | ONNX (quantized) |

### Code Difference

The only difference is in initialization:

```python
# PyTorch
ner = BusNERInference("models/bus_ner_transformer", use_onnx=False)

# ONNX
ner = BusNERInference("models/bus_ner_onnx", use_onnx=True)
```

After initialization, usage is identical:

```python
result = ner.extract("AC bus from Bangalore to Mumbai")
```

---

## Summary

The inference pipeline follows these steps:

| Step | Function | What Happens |
|------|----------|--------------|
| 1 | `__init__` | Load model (PyTorch or ONNX) |
| 2 | `extract()` | Entry point for extraction |
| 3 | `_predict()` | Tokenize → Forward → Decode |
| 4 | BIO decoding | Collect B/I sequences |
| 5 | Format output | Create entity dictionary |

**Key insight:** The model predicts BIO tags for each subword token. We only use predictions from the first subword of each word, then decode B/I sequences into entity spans.
