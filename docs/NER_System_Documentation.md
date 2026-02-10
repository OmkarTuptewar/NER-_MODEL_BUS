# Bus Domain NER System - Technical Documentation

A comprehensive guide to the Transformer-based Named Entity Recognition (NER) system for domain-specific query understanding in the bus/travel domain.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Why NER is Needed for This Project](#why-ner-is-needed-for-this-project)
4. [Basic NLP Concepts](#basic-nlp-concepts)
5. [What is NER (Named Entity Recognition)](#what-is-ner-named-entity-recognition)
6. [Labeling Strategies](#labeling-strategies)
7. [Why BIO Was Chosen](#why-bio-was-chosen)
8. [NER Approaches Comparison](#ner-approaches-comparison)
9. [Why MiniLM Was Chosen](#why-minilm-was-chosen)
10. [Dataset Creation Strategy](#dataset-creation-strategy)
11. [Architecture Overview](#architecture-overview)
12. [Detailed System Architecture](#detailed-system-architecture)
13. [Training Pipeline](#training-pipeline)
14. [Inference Pipeline](#inference-pipeline)
15. [Tokenization and Subword Handling](#tokenization-and-subword-handling)
16. [Inside the MiniLM Model](#inside-the-minilm-model)
17. [API Design (FastAPI)](#api-design-fastapi)
18. [Important Code Walkthroughs](#important-code-walkthroughs)
19. [Performance, Scalability, and Cost](#performance-scalability-and-cost)
20. [Alternative Designs and Trade-offs](#alternative-designs-and-trade-offs)
21. [Limitations](#limitations)
22. [Future Improvements](#future-improvements)
23. [Interview Preparation](#interview-preparation)
24. [Conclusion](#conclusion)

---

## Project Overview

This project implements a production-ready Named Entity Recognition (NER) system designed specifically for parsing natural language bus search queries. The system uses a fine-tuned MiniLM transformer model to extract structured travel information from unstructured user input.

**Core Technologies:**
- **Model**: MiniLM-L12-H384 (distilled transformer)
- **Framework**: Hugging Face Transformers + PyTorch
- **API**: FastAPI with async support
- **Optimization**: ONNX Runtime for production inference

**Entity Types Supported:**
| Entity | Description | Example |
|--------|-------------|---------|
| SRC | Source city | "Bangalore", "Mumbai" |
| DEST | Destination city | "Chennai", "Delhi" |
| BUS_TYPE | AC or Non-AC | "AC", "non-ac" |
| SEAT_TYPE | Sleeper or Seater | "sleeper", "semi sleeper" |
| TIME | Time of travel | "morning", "10 pm" |
| DATE | Date of travel | "tomorrow", "next Friday" |
| OPERATOR | Bus operator | "VRL", "KSRTC" |
| BOARDING_POINT | Pickup location | "Majestic", "Electronic City" |
| DROPPING_POINT | Drop location | "Hitec City", "Koyambedu" |

---

## Problem Statement

When users search for bus tickets, they express their intent in natural language:

```
"AC sleeper bus from Bangalore to Mumbai tomorrow morning"
```

To process this query programmatically, we need to extract structured data:

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

**Challenges:**
1. Users write queries in many different ways
2. Entity values can be multi-word ("Electronic City", "day after tomorrow")
3. Context matters ("from X to Y" vs just mentioning cities)
4. Queries may contain only partial information
5. Spelling variations and case differences

---

## Why NER is Needed for This Project

Traditional approaches like keyword matching or regex fail because:

| Approach | Limitation |
|----------|------------|
| Keyword matching | Cannot distinguish SRC from DEST |
| Regex patterns | Brittle; breaks with new formats |
| Slot filling | Requires strict query structure |
| Intent classification | Only identifies intent, not entities |

**NER solves these problems by:**
- Understanding context to differentiate entities
- Handling variable-length entities
- Generalizing to unseen query patterns
- Working with incomplete or ambiguous input

---

## Basic NLP Concepts

### What are Embeddings?

Embeddings are numerical representations (vectors) of words or tokens that capture semantic meaning. Words with similar meanings have similar vectors.

```
"bus"     → [0.2, -0.5, 0.8, ...]
"coach"   → [0.3, -0.4, 0.7, ...]  ← similar to "bus"
"airport" → [-0.1, 0.9, 0.2, ...]  ← different
```

### Static vs Contextual Embeddings

| Type | Description | Example |
|------|-------------|---------|
| **Static** (Word2Vec, GloVe) | Same vector regardless of context | "bank" has one embedding |
| **Contextual** (BERT, MiniLM) | Vector depends on surrounding words | "river bank" vs "bank account" |

MiniLM uses **contextual embeddings**, which is critical for NER where the same word can be SRC or DEST depending on context.

### How Transformers Understand Context

Transformers use **self-attention** to weigh the importance of each word relative to every other word in a sentence.

For the query "bus from Bangalore to Mumbai":
- "Bangalore" attends to "from" → likely SRC
- "Mumbai" attends to "to" → likely DEST

This bidirectional attention allows the model to understand relationships regardless of word position.

### What are Attention Masks and Token IDs?

When we tokenize text for a transformer:

- **Token IDs**: Numerical indices for each token in the vocabulary
- **Attention Mask**: Binary mask (1 = real token, 0 = padding)


```python
# Example tokenization
text = "bus from Bangalore"
token_ids = [2187, 2013, 14411]  # Vocabulary indices
attention_mask = [1, 1, 1]       # All real tokens
```

---

## What is NER (Named Entity Recognition)

NER is a sequence labeling task that assigns a category label to each token in a sentence.

**Input**: "AC bus from Bangalore to Mumbai"

**Output**:
| Token | Label |
|-------|-------|
| AC | B-BUS_TYPE |
| bus | O |
| from | O |
| Bangalore | B-SRC |
| to | O |
| Mumbai | B-DEST |

NER differs from classification because:
- It operates at the **token level**, not sentence level
- It must handle **entity boundaries** (where entities start and end)
- Multiple entities of different types can exist in one sentence

---

## Labeling Strategies

### BIO (Begin, Inside, Outside)

The most common NER labeling scheme:
- **B-X**: Beginning of entity type X
- **I-X**: Inside (continuation) of entity type X
- **O**: Outside any entity

Example: "Electronic City"
| Token | Label |
|-------|-------|
| Electronic | B-BOARDING_POINT |
| City | I-BOARDING_POINT |

### BILOU (Begin, Inside, Last, Outside, Unit)

More granular than BIO:
- **B-X**: Beginning of multi-token entity
- **I-X**: Inside entity
- **L-X**: Last token of entity
- **U-X**: Unit (single-token entity)
- **O**: Outside

### IO (Inside, Outside)

Simplest scheme - no begin markers:
- **I-X**: Token is part of entity X
- **O**: Outside


### Comparison Table

| Scheme | Labels per Entity | Boundary Precision | Complexity |
|--------|-------------------|-------------------|------------|
| **IO** | 1 | Cannot detect adjacent entities | Low |
| **BIO** | 2 | Detects start of entities | Medium |
| **BILOU** | 4 | Full boundary detection | High |

---

## Why BIO Was Chosen

We chose BIO for this project because:

1. **Sufficient for our domain**: Adjacent entities of the same type are rare in bus queries
2. **Fewer labels = easier training**: 19 labels (O + 9×2) vs 37 for BILOU
3. **Industry standard**: Most NER datasets and models use BIO
4. **Balanced complexity**: More expressive than IO, simpler than BILOU

Our label mapping from `label2id.json`:
```json
{
  "O": 0,
  "B-SRC": 1, "I-SRC": 2,
  "B-DEST": 3, "I-DEST": 4,
  "B-BUS_TYPE": 5, "I-BUS_TYPE": 6,
  "B-SEAT_TYPE": 7, "I-SEAT_TYPE": 8,
  "B-TIME": 9, "I-TIME": 10,
  "B-DATE": 11, "I-DATE": 12,
  "B-OPERATOR": 13, "I-OPERATOR": 14,
  "B-BOARDING_POINT": 15, "I-BOARDING_POINT": 16,
  "B-DROPPING_POINT": 17, "I-DROPPING_POINT": 18
}
```

---

## NER Approaches Comparison

### Rule-Based NER

Uses handcrafted rules and gazetteers (entity lists).

```python
# Example rule-based approach
if "from" in query and "to" in query:
    src = extract_word_after("from")
    dest = extract_word_after("to")
```

**Pros**: Fast, explainable, no training needed  
**Cons**: Brittle, doesn't generalize, maintenance nightmare

### spaCy CNN-Based NER

Uses convolutional neural networks with transition-based parsing.

**Pros**: Fast training, small model size (~10MB)  
**Cons**: Limited context window, struggles with complex patterns

### Transformer-Based NER

Uses attention mechanisms for full sequence understanding.

**Pros**: Best accuracy, handles complex context, transfer learning  
**Cons**: Larger models, slower without optimization

### Comparison Table

| Aspect | Rule-Based | spaCy CNN | Transformer |
|--------|------------|-----------|-------------|
| **Accuracy** | Low-Medium | Medium-High | Highest |
| **Speed** | Fastest | Fast | Medium |
| **Model Size** | N/A | 5-15 MB | 50-150 MB |
| **Training Data Needed** | None | Moderate | Moderate |
| **Generalization** | Poor | Good | Excellent |
| **Context Understanding** | None | Local | Global |
| **Maintenance** | High | Low | Low |

---

## Why MiniLM Was Chosen

### What is MiniLM?

MiniLM is a **distilled** version of BERT, meaning it was trained to mimic a larger teacher model while being significantly smaller and faster.


**Model Specifications** (`microsoft/MiniLM-L12-H384-uncased`):
- 12 transformer layers
- 384 hidden dimensions
- ~22M parameters (vs BERT's 110M)
- Vocabulary: 30,522 tokens (WordPiece)

### What is Knowledge Distillation?

Distillation transfers knowledge from a large "teacher" model to a smaller "student" model:

1. Teacher (BERT-large) makes predictions
2. Student (MiniLM) learns to match teacher's outputs
3. Student becomes smaller but retains most capability

### Comparison: MiniLM vs BERT vs spaCy

| Metric | spaCy CNN | MiniLM | BERT-base |
|--------|-----------|--------|-----------|
| **Parameters** | ~5M | ~22M | ~110M |
| **Inference Time** | ~2ms | ~10-20ms | ~40-60ms |
| **Model Size** | 10 MB | 50 MB | 440 MB |
| **F1 Score (NER)** | 85-88% | 90-93% | 91-94% |
| **Context Window** | Limited | 512 tokens | 512 tokens |
| **Pre-training** | Word vectors | MLM + Distillation | MLM |

**Why MiniLM for this project:**
1. **6x smaller than BERT** with <2% accuracy loss
2. **Fast inference** suitable for real-time APIs
3. **ONNX-compatible** for production optimization
4. **Excellent transfer learning** from massive pre-training

---

## Dataset Creation Strategy

### Template-Based Generation

We use a **synthetic data generation** approach with templates in `data_generator.py`.

**Templates** define query patterns with placeholders:
```python
TEMPLATES = [
    "bus from {SRC} to {DEST}",
    "{BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE}",
    "{OPERATOR} bus from {SRC} to {DEST} pickup at {BOARDING_POINT}",
    # ... 100+ templates
]
```

**Entity pools** provide values:
```python
ENTITY_VALUES = {
    "SRC": ["Bangalore", "Mumbai", "Chennai", ...],
    "DEST": ["Bangalore", "Mumbai", "Chennai", ...],
    "BUS_TYPE": ["AC", "Non-AC"],
    # ...
}
```

### Why Template-Based Generation?

| Approach | Pros | Cons |
|----------|------|------|
| Manual annotation | High quality | Expensive, slow |
| Template generation | Scalable, cheap | Limited variety |
| Weak supervision | Semi-automatic | Noisy labels |

We chose templates because:
1. **Perfect labels**: No annotation errors
2. **Scalable**: Can generate millions of samples
3. **Controllable**: Easy to add new patterns
4. **Domain-specific**: Tailored to bus queries

### BIO Tag Generation

The generator automatically computes BIO tags from span annotations:

```python
# Input: filled template with entity positions
text = "AC bus from Bangalore to Mumbai"
entities = [(0, 2, "BUS_TYPE"), (14, 23, "SRC"), (27, 33, "DEST")]

# Output: BIO-tagged data
{
    "tokens": ["AC", "bus", "from", "Bangalore", "to", "Mumbai"],
    "ner_tags": [5, 0, 0, 1, 0, 3]  # B-BUS_TYPE, O, O, B-SRC, O, B-DEST
}
```

---

## Architecture Overview

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Templates + Entities  →  data_generator.py  →  BIO JSON Data   │
│                                    │                             │
│                                    ▼                             │
│  BIO JSON Data  →  train_ner.py  →  Fine-tuned MiniLM Model     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE PHASE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Query  →  Tokenizer  →  MiniLM  →  BIO Tags  →  Entities  │
│       ↑                                                    │     │
│       └──────────────  FastAPI (api.py)  ←─────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| Data Generator | `data_generator.py` | Create synthetic training data |
| Trainer | `train_ner.py` | Fine-tune MiniLM on BIO data |
| Inference | `inference.py` | Run predictions, extract entities |
| API | `api.py` | HTTP endpoints for NER service |
| ONNX Export | `export_onnx.py` | Optimize model for production |

---

## Detailed System Architecture

### Data Layer

**Files**: `data/training_data.json`, `data/training_data_bio.json`

The data layer consists of:
1. **Span-format data**: Text with character-level entity positions
2. **BIO-format data**: Tokenized with per-token labels


### Preprocessing Layer

**Tokenization Flow**:
```
"AC sleeper from Bangalore"
        ↓ (whitespace split)
["AC", "sleeper", "from", "Bangalore"]
        ↓ (WordPiece tokenization)
["ac", "sleep", "##er", "from", "bangalore"]
        ↓ (with special tokens)
["[CLS]", "ac", "sleep", "##er", "from", "bangalore", "[SEP]"]
```

### Model Layer

The MiniLM model with a token classification head:
- **Input**: Token IDs, attention mask
- **Output**: Logits for each token × 19 labels

### Post-Processing Layer

BIO tags are converted back to entities:
1. Collect consecutive B-X and I-X tokens
2. Join tokens to form entity text
3. Map to structured output format

### API Layer

FastAPI provides:
- `/ner/extract` - Single query extraction
- `/ner/extract/detailed` - With character positions
- `/ner/batch` - Batch processing
- `/health` - Service health check

---

## Training Pipeline

### Step 1: Load BIO Data

```python
def load_bio_data(data_path: str) -> List[Dict]:
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data
```

### Step 2: Prepare HuggingFace Datasets

```python
def prepare_dataset(data, test_split=0.1):
    dataset = Dataset.from_dict({
        "tokens": [s["tokens"] for s in data],
        "ner_tags": [s["ner_tags"] for s in data],
    })
    return dataset.train_test_split(test_size=test_split)
```

### Step 3: Tokenize and Align Labels

This is the critical step - aligning BIO labels with subword tokens:

```python
def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length"
    )
    
    labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_sequence = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                label_sequence.append(-100)
            elif word_idx != previous_word_idx:
                # First subword gets the actual label
                label_sequence.append(label_ids[word_idx])
            else:
                # Subsequent subwords get -100
                label_sequence.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_sequence)
    
    tokenized["labels"] = labels
    return tokenized
```

### Step 4: Create Model

```python
model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/MiniLM-L12-H384-uncased",
    num_labels=19,
    label2id=label2id,
    id2label=id2label
)
```

### Step 5: Train with HuggingFace Trainer

```python
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

### Metrics

We use **seqeval** for entity-level metrics:
- **Precision**: Of entities predicted, how many are correct?
- **Recall**: Of actual entities, how many were found?
- **F1 Score**: Harmonic mean of precision and recall

---

## Inference Pipeline

### BusNERInference Class

The main inference class in `inference.py`:

```python
class BusNERInference:
    def __init__(self, model_path: str, use_onnx: bool = False):
        if use_onnx:
            self.tokenizer, self.model, ... = self._load_onnx_model()
        else:
            self.tokenizer, self.model, ... = self._load_pytorch_model()
    
    def extract(self, query: str) -> Dict[str, Optional[str]]:
        entities = self._predict(query)
        result = {etype: None for etype in ALL_ENTITY_TYPES}
        for entity_text, label, _, _ in entities:
            if result[label] is None:
                result[label] = entity_text
        return result
```

### Prediction Flow

1. **Tokenize**: Split by whitespace, then WordPiece
2. **Forward pass**: Get logits from model
3. **Argmax**: Select highest probability label per token
4. **Align**: Map subword predictions back to words
5. **Extract**: Collect B/I sequences into entities

---

## Tokenization and Subword Handling

### The Subword Challenge

WordPiece may split words into subwords:
```
"Bangalore" → ["ban", "##galore"]  (2 tokens)
"KSRTC"     → ["ks", "##rt", "##c"] (3 tokens)
```

### Label Alignment Strategy

We use the **first subword only** strategy:
- First subword gets the word's actual label
- Subsequent subwords get `-100` (ignored in loss)

Why this works:
1. Training focuses on predicting the first subword correctly
2. Inference extracts labels from first subwords only
3. No confusion from multiple predictions per word

### Example

```
Word:     "Bangalore"
Subwords: ["ban", "##galore"]
Labels:   [B-SRC, -100]  ← Only first subword labeled
```

---

## Inside the MiniLM Model

### Transformer Architecture

MiniLM consists of:
1. **Embedding Layer**: Token embeddings + position embeddings
2. **12 Transformer Layers**: Each with self-attention and feed-forward
3. **Classification Head**: Linear layer (768 → 19 labels)

### Self-Attention Mechanism

For each token, attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q (Query): What am I looking for?
- K (Key): What do I contain?
- V (Value): What do I represent?

### Why MiniLM is Efficient

MiniLM uses **deep self-attention distillation**:
- Matches the attention patterns of BERT-large
- Focuses on the most informative attention heads
- Results in smaller model with similar attention behavior

### Model Dimensions

| Component | Dimension |
|-----------|-----------|
| Hidden size | 384 |
| Intermediate size | 1536 |
| Attention heads | 12 |
| Layers | 12 |
| Vocabulary | 30,522 |
| Max sequence | 512 |

---

## API Design (FastAPI)

### Why FastAPI?

| Feature | Benefit |
|---------|---------|
| Async support | High concurrency |
| Auto documentation | Swagger UI built-in |
| Pydantic models | Request/response validation |
| Type hints | Better IDE support |
| Performance | One of fastest Python frameworks |

### Lifespan Management

Model loading happens once at startup:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_inference
    ner_inference = BusNERInference(MODEL_PATH, use_onnx=True)
    yield
    ner_inference = None
```

### Endpoints

**POST /ner/extract**
```json
// Request
{"query": "AC bus from Bangalore to Mumbai tomorrow"}

// Response
{
  "query": "AC bus from Bangalore to Mumbai tomorrow",
  "entities": {
    "SRC": "Bangalore",
    "DEST": "Mumbai",
    "BUS_TYPE": "AC",
    "DATE": "tomorrow"
  }
}
```

**POST /ner/batch**
```json
// Request
{"queries": ["bus to Mumbai", "sleeper from Delhi"]}

// Response
{"results": [...]}
```

---

## Important Code Walkthroughs

### 1. BIO Conversion (data_generator.py)

```python
def convert_span_to_bio(sample, tokenizer=None):
    text = sample["text"]
    entities = sample["entities"]  # [[start, end, label], ...]
    
    tokens = text.split()
    
    # Build char→token mapping
    char_to_token = {}
    current_char = 0
    for token_idx, token in enumerate(tokens):
        for i in range(len(token)):
            char_to_token[current_char + i] = token_idx
        current_char += len(token) + 1
    
    # Initialize all as "O"
    ner_tags = [label2id["O"]] * len(tokens)
    
    # Assign B/I tags
    for start, end, label in entities:
        entity_tokens = set()
        for char_idx in range(start, end):
            if char_idx in char_to_token:
                entity_tokens.add(char_to_token[char_idx])
        
        sorted_indices = sorted(entity_tokens)
        for i, token_idx in enumerate(sorted_indices):
            if i == 0:
                ner_tags[token_idx] = label2id[f"B-{label}"]
            else:
                ner_tags[token_idx] = label2id[f"I-{label}"]
    
    return {"tokens": tokens, "ner_tags": ner_tags}
```

**Why this matters**: This function bridges character-level spans to token-level BIO tags, which is essential for training.

### 2. Subword Label Alignment (train_ner.py)

```python
for word_idx in word_ids:
    if word_idx is None:
        label_sequence.append(-100)  # Special tokens
    elif word_idx != previous_word_idx:
        label_sequence.append(label_ids[word_idx])  # First subword
    else:
        label_sequence.append(-100)  # Subsequent subwords
    previous_word_idx = word_idx
```

**Why this matters**: Without this, the model would try to predict labels for padding and continuation tokens, degrading performance.

### 3. Entity Extraction (inference.py)

```python
for word_idx, label in word_labels:
    if label.startswith("B-"):
        # Save previous entity
        if current_entity_words:
            entities.append((entity_text, current_label, ...))
        # Start new entity
        current_entity_words = [words[word_idx]]
        current_label = label[2:]  # Remove "B-"
        
    elif label.startswith("I-") and current_label == label[2:]:
        # Continue current entity
        current_entity_words.append(words[word_idx])
        
    else:  # "O" or mismatched I-
        # End current entity
        if current_entity_words:
            entities.append((entity_text, current_label, ...))
        current_entity_words = []
```

**Why this matters**: This state machine correctly handles multi-word entities and entity boundaries.

### 4. FastAPI Model Loading (api.py)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_inference
    try:
        ner_inference = BusNERInference(MODEL_PATH, use_onnx=True)
        print("Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
    yield
    ner_inference = None
```

**Why this matters**: Loading happens once at startup (not per-request), ensuring low latency. Error handling allows graceful degradation.

---

## Performance, Scalability, and Cost

### Latency Optimization

| Optimization | Speedup | Trade-off |
|--------------|---------|-----------|
| ONNX Runtime | 2-5x | Additional export step |
| INT8 Quantization | 1.5-2x | Slight accuracy loss |
| Batch processing | 3-10x | Increased memory |
| GPU inference | 5-20x | Hardware cost |

### ONNX Export

```python
from optimum.onnxruntime import ORTModelForTokenClassification

ort_model = ORTModelForTokenClassification.from_pretrained(
    model_dir,
    export=True
)
ort_model.save_pretrained(output_dir)
```

### Scalability Considerations

| Aspect | Recommendation |
|--------|----------------|
| Horizontal scaling | Run multiple API instances behind load balancer |
| Caching | Cache results for repeated queries |
| Async processing | Use FastAPI async endpoints |
| Batch API | Process multiple queries per request |

### Cost Analysis

| Component | Monthly Cost (AWS) |
|-----------|-------------------|
| t3.medium (CPU) | ~$30 |
| g4dn.xlarge (GPU) | ~$400 |
| Model storage (S3) | ~$5 |

For most use cases, **CPU with ONNX** provides the best cost-performance ratio.

---

## Alternative Designs and Trade-offs

### Pure spaCy System

**Architecture**: spaCy's built-in NER with custom training

**Pros**:
- Simpler setup
- Faster training
- Smaller models

**Cons**:
- Lower accuracy on complex patterns
- Limited context understanding

### Rule-Based System

**Architecture**: Pattern matching with entity lists

**Pros**:
- No training needed
- Fully explainable
- Instant updates

**Cons**:
- Doesn't generalize
- Maintenance burden grows
- Cannot handle variations

### LLM-Based Extraction

**Architecture**: GPT-4 or similar with prompting

**Pros**:
- Zero-shot capability
- Handles complex queries
- No training needed

**Cons**:
- High latency (seconds)
- Expensive ($0.01+ per query)
- No local deployment

### Hybrid Approach

**Architecture**: Rules for common patterns + ML for edge cases

**Pros**:
- Fast for simple queries
- Fallback for complex cases

**Cons**:
- System complexity
- Two systems to maintain

---

## Limitations

### Current System Limitations

1. **English only**: No multilingual support
2. **Fixed entity types**: Cannot dynamically add new types
3. **No confidence scores**: All-or-nothing extraction
4. **Single-value entities**: Takes first occurrence only
5. **Training data bias**: Limited to template patterns

### Edge Cases Not Handled

- Misspellings ("Banglore" → "Bangalore")
- Abbreviations ("blr" → "Bangalore")
- Code-switching (mixing languages)
- Very long queries (>512 tokens)

---

## Future Improvements

### Short-term

1. **Add confidence filtering**: Output probability scores with entities
2. **Handle multiple values**: Return lists for repeated entities
3. **Spelling correction**: Pre-process with fuzzy matching

### Medium-term

1. **Active learning**: Collect and label edge cases
2. **Multi-lingual support**: Fine-tune on translated data
3. **Ensemble models**: Combine multiple model predictions

### Long-term

1. **Self-training loop**: Automatically improve from production data
2. **Entity linking**: Map entities to canonical database IDs
3. **Conversational NER**: Handle multi-turn dialogues

---

## Interview Preparation

### Questions This Project Prepares You For

**NLP Fundamentals**:
- What is NER and how does it differ from text classification?
- Explain the BIO tagging scheme.
- What are contextual vs static embeddings?

**Transformers**:
- How does self-attention work?
- What is knowledge distillation?
- Why use MiniLM over BERT?

**System Design**:
- How would you scale this NER service to 10K QPS?
- What's your strategy for handling model updates in production?
- How do you monitor NER quality in production?

**ML Engineering**:
- How do you handle subword tokenization in NER?
- What metrics do you use for NER evaluation?
- How would you generate training data without manual annotation?

### Sample Interview Answers

**Q: Why did you choose MiniLM over BERT?**

A: MiniLM offers 99% of BERT's accuracy at 1/5th the size and 3x the speed. For a production API with latency requirements, this trade-off is optimal. The distillation process preserves attention patterns while reducing parameters.

**Q: How does your system handle multi-word entities?**

A: We use BIO tagging where B- marks entity start and I- marks continuation. During inference, we collect consecutive B/I tokens of the same type into a single entity. The key is proper handling during training via label alignment with subword tokens.

---

## Conclusion

This NER system demonstrates production-ready machine learning engineering:

1. **Synthetic data generation** eliminates manual annotation costs
2. **MiniLM transformer** balances accuracy and efficiency
3. **BIO tagging** handles variable-length entities
4. **ONNX optimization** enables sub-20ms inference
5. **FastAPI** provides a scalable HTTP interface

The architecture is suitable for:
- Real-time bus booking applications
- Similar domain-specific entity extraction tasks
- Learning transformer-based NLP systems

**Key Takeaways**:
- Start with simpler models (MiniLM, not GPT-4) for production
- Invest in data quality over model complexity
- Optimize inference early (ONNX, batching)
- Design APIs for the actual use case (single + batch)

---

*Documentation generated based on the actual codebase. For implementation details, refer to the source files in `src/`.*
