# ONNX Export Process - Step by Step Guide

This document explains in detail how to export the trained PyTorch NER model to ONNX format using `src/export_onnx.py`. We'll cover what ONNX is, why it's beneficial, and walk through each step of the export process.

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is ONNX?](#what-is-onnx)
3. [Benefits of ONNX](#benefits-of-onnx)
4. [Export Flow Overview](#export-flow-overview)
5. [Function Explanations](#function-explanations)
6. [Quantization](#quantization)
7. [Output Files](#output-files)
8. [Verification](#verification)
9. [Usage Examples](#usage-examples)

---

## Introduction

### Why Export to ONNX?

After training your NER model with PyTorch, you have a `.safetensors` file that requires the full PyTorch library to run. For production deployment, this has drawbacks:

| Issue | Impact |
|-------|--------|
| Large dependencies | PyTorch is ~2GB installed |
| Slow cold start | Loading PyTorch takes seconds |
| CPU inefficiency | PyTorch not optimized for CPU inference |
| Limited platforms | Python-only deployment |

**ONNX solves these problems** by providing a standardized, optimized model format.

### Prerequisites

```bash
pip install optimum[onnxruntime]
```

---

## What is ONNX?

**ONNX** (Open Neural Network Exchange) is an open format for representing machine learning models.

```
┌─────────────────────────────────────────────────────────────────┐
│                        ONNX ECOSYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Training Frameworks         ONNX Format        Runtime         │
│   ─────────────────          ───────────        ───────          │
│                                                                  │
│   PyTorch      ──┐                           ┌── ONNX Runtime    │
│   TensorFlow   ──┼──→  model.onnx  ──────────┼── TensorRT        │
│   JAX          ──┤     (portable)            ├── OpenVINO        │
│   Keras        ──┘                           ├── CoreML          │
│                                              └── DirectML        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key idea:** Train once in any framework, deploy anywhere.

---

## Benefits of ONNX

### Performance Comparison

| Metric | PyTorch | ONNX Runtime |
|--------|---------|--------------|
| **Inference latency** | ~30ms | ~10ms |
| **Speedup** | 1x | 2-5x |
| **Memory usage** | Higher | Lower |
| **CPU optimization** | Basic | AVX2/AVX512 optimized |

### Feature Comparison

| Feature | PyTorch | ONNX |
|---------|---------|------|
| Cross-platform | Python only | C++, Java, JS, C#, etc. |
| Model size | ~50 MB | ~50 MB (or ~15 MB quantized) |
| GPU support | Yes | Yes (CUDA EP) |
| Edge deployment | Difficult | Easy |
| Quantization | Complex | Built-in |

### When to Use ONNX

| Scenario | Recommendation |
|----------|----------------|
| Development | PyTorch |
| Production API | ONNX |
| Mobile deployment | ONNX |
| Browser inference | ONNX.js |
| Embedded systems | ONNX (quantized) |

---

## Export Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ONNX EXPORT PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

  PyTorch Model Directory
  models/bus_ner_transformer/
  ├── model.safetensors
  ├── config.json
  ├── tokenizer.json
  ├── label2id.json
  └── id2label.json
            │
            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 1: LOAD TOKENIZER                                     │
  │  tokenizer = AutoTokenizer.from_pretrained(model_dir)       │
  └─────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 2: EXPORT TO ONNX                                     │
  │  ort_model = ORTModelForTokenClassification.from_pretrained(│
  │      model_dir, export=True                                 │
  │  )                                                          │
  │  - Traces model with sample inputs                          │
  │  - Converts operations to ONNX operators                    │
  │  - Optimizes graph                                          │
  └─────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 3: SAVE MODEL                                         │
  │  ort_model.save_pretrained(output_dir)                      │
  │  tokenizer.save_pretrained(output_dir)                      │
  └─────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Step 4: COPY LABEL MAPPINGS                                │
  │  shutil.copy(label2id.json)                                 │
  │  shutil.copy(id2label.json)                                 │
  └─────────────────────────────────────────────────────────────┘
            │
            ▼
  ONNX Model Directory
  models/bus_ner_onnx/
  ├── model.onnx          ← ONNX graph
  ├── config.json
  ├── tokenizer.json
  ├── label2id.json
  └── id2label.json
```

---

## Function Explanations

### Function: `export_to_onnx(model_path, output_path, quantize=False)`

**What it does:** Converts a PyTorch model to ONNX format.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | str | Path to trained PyTorch model |
| `output_path` | str | Path to save ONNX model |
| `quantize` | bool | Apply INT8 quantization |

### Step-by-Step Breakdown

#### Step 1: Load Tokenizer

```python
print("[Step 1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
```

The tokenizer is needed because it will be saved with the ONNX model.

#### Step 2: Export to ONNX

```python
print("[Step 2/4] Exporting model to ONNX...")
ort_model = ORTModelForTokenClassification.from_pretrained(
    model_dir,
    export=True  # This triggers ONNX export
)
```

**What happens internally:**
1. Load PyTorch model
2. Create sample inputs (dummy tokens)
3. Trace the model execution
4. Convert PyTorch operations to ONNX operators
5. Apply graph optimizations

#### Step 3: Save ONNX Model

```python
print("[Step 3/4] Saving ONNX model...")
output_dir.mkdir(parents=True, exist_ok=True)
ort_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

This saves:
- `model.onnx` - The ONNX graph
- Tokenizer files
- Config files

#### Step 4: Copy Label Mappings

```python
print("[Step 4/4] Copying label mappings...")
for label_file in ["label2id.json", "id2label.json"]:
    src = model_dir / label_file
    dst = output_dir / label_file
    if src.exists():
        shutil.copy(src, dst)
```

These files are needed during inference to convert prediction IDs back to label names.

---

## Quantization

### What is Quantization?

Quantization reduces model precision from FP32 (32-bit floats) to INT8 (8-bit integers).

```
┌─────────────────────────────────────────────────────────────┐
│                     QUANTIZATION                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   FP32 Weight: 0.12345678                                   │
│        ↓                                                     │
│   INT8 Weight: 31  (scaled integer)                         │
│                                                              │
│   Benefits:                                                  │
│   - 4x smaller model size                                   │
│   - 2-4x faster inference                                   │
│   - Lower memory bandwidth                                  │
│                                                              │
│   Trade-off:                                                 │
│   - Slight accuracy loss (usually <1%)                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Quantization Code

```python
if quantize:
    print("[Optional] Applying INT8 quantization...")
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    
    quantizer = ORTQuantizer.from_pretrained(output_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
    
    quantizer.quantize(
        save_dir=output_dir,
        quantization_config=qconfig
    )
```

### Quantization Options

| Config | Target Hardware | Description |
|--------|-----------------|-------------|
| `avx2` | Intel CPUs (2013+) | Good balance |
| `avx512_vnni` | Intel CPUs (2019+) | Best for server CPUs |
| `arm64` | ARM processors | Mobile/embedded |

### Size Comparison

| Model | Size |
|-------|------|
| PyTorch (FP32) | ~50 MB |
| ONNX (FP32) | ~50 MB |
| ONNX (INT8 quantized) | ~15 MB |

---

## Output Files

After export, the output directory contains:

```
models/bus_ner_onnx/
├── model.onnx              # 50 MB - ONNX model graph
├── config.json             # Model configuration
├── tokenizer.json          # Fast tokenizer data
├── tokenizer_config.json   # Tokenizer settings
├── special_tokens_map.json # Special token definitions
├── vocab.txt               # Vocabulary
├── label2id.json           # Label → ID mapping
└── id2label.json           # ID → Label mapping
```

### File Descriptions

| File | Purpose | Size |
|------|---------|------|
| `model.onnx` | The neural network graph | ~50 MB |
| `config.json` | Model architecture config | ~1 KB |
| `tokenizer.json` | Tokenizer vocabulary and rules | ~700 KB |
| `label2id.json` | Maps "B-SRC" → 1, etc. | ~500 B |
| `id2label.json` | Maps 1 → "B-SRC", etc. | ~500 B |

---

## Verification

### Function: `verify_onnx_model(onnx_path)`

**What it does:** Tests the exported ONNX model with sample queries.

```python
def verify_onnx_model(onnx_path):
    from optimum.onnxruntime import ORTModelForTokenClassification
    
    onnx_dir = Path(onnx_path)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
    model = ORTModelForTokenClassification.from_pretrained(onnx_dir)
    
    # Load label mappings
    with open(onnx_dir / "id2label.json", 'r') as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    
    # Test queries
    test_queries = [
        "AC sleeper bus from Bangalore to Mumbai tomorrow",
        "VRL Non-AC from Chennai to Hyderabad next Friday"
    ]
    
    for query in test_queries:
        tokens = query.split()
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=2)[0]
        
        # Print detected entities
        ...
```

### Example Verification Output

```
VERIFYING ONNX MODEL
============================================================

Test Results:

Query: "AC sleeper bus from Bangalore to Mumbai tomorrow"
  - B-BUS_TYPE: "AC"
  - B-SEAT_TYPE: "sleeper"
  - B-SRC: "Bangalore"
  - B-DEST: "Mumbai"
  - B-DATE: "tomorrow"

Query: "VRL Non-AC from Chennai to Hyderabad next Friday"
  - B-OPERATOR: "VRL"
  - B-BUS_TYPE: "Non-AC"
  - B-SRC: "Chennai"
  - B-DEST: "Hyderabad"
  - B-DATE: "next"

VERIFICATION COMPLETE
============================================================
```

---

## Usage Examples

### Basic Export

```bash
python src/export_onnx.py
```

Uses default paths:
- Input: `models/bus_ner_transformer`
- Output: `models/bus_ner_onnx`

### Export with Custom Paths

```bash
python src/export_onnx.py \
    --model models/my_trained_model \
    --output models/my_onnx_model
```

### Export with Quantization

```bash
python src/export_onnx.py --quantize
```

Creates an additional quantized model for smaller size.

### Export and Verify

```bash
python src/export_onnx.py --verify
```

Runs test queries after export to ensure the model works.

### Full Command

```bash
python src/export_onnx.py \
    --model models/bus_ner_transformer \
    --output models/bus_ner_onnx \
    --quantize \
    --verify
```

### Using the Exported Model

```python
from inference import BusNERInference

# Load ONNX model
ner = BusNERInference("models/bus_ner_onnx", use_onnx=True)

# Extract entities (same API as PyTorch)
result = ner.extract("AC sleeper from Bangalore to Mumbai")
print(result)
```

Output:
```python
{
    "SRC": "Bangalore",
    "DEST": "Mumbai",
    "BUS_TYPE": "AC",
    "SEAT_TYPE": "sleeper",
    ...
}
```

---

## Summary

The ONNX export process follows these steps:

| Step | Action | Result |
|------|--------|--------|
| 1 | Load tokenizer | Tokenizer ready |
| 2 | Export to ONNX | `model.onnx` created |
| 3 | Save model | All files saved |
| 4 | Copy mappings | Labels available for inference |
| 5 | (Optional) Quantize | Smaller, faster model |
| 6 | (Optional) Verify | Test with sample queries |

**Key benefits of ONNX:**
- 2-5x faster inference
- Cross-platform deployment
- Smaller dependencies
- Optional quantization for edge devices

After export, the ONNX model is used identically to the PyTorch model through the `BusNERInference` class.
