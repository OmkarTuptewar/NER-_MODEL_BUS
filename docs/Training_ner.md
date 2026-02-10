# NER Model Training Process - Step by Step Guide

This document explains in detail how the Bus NER model is trained using `src/train_ner.py`. We'll walk through each step of the training pipeline, explain what each function does, and show exactly what the data looks like at each stage.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Training Flow Overview](#training-flow-overview)
3. [Step 1: Load BIO Data](#step-1-load-bio-data)
4. [Step 2: Prepare Dataset](#step-2-prepare-dataset)
5. [Step 3: Tokenize and Align Labels](#step-3-tokenize-and-align-labels)
6. [Step 4: Create Model](#step-4-create-model)
7. [Step 5: Train Model](#step-5-train-model)
8. [Step 6: Save Model](#step-6-save-model)
9. [Understanding Subword Label Alignment](#understanding-subword-label-alignment)
10. [Training Configuration Explained](#training-configuration-explained)
11. [Metrics Computation](#metrics-computation)
12. [Model Architecture](#model-architecture)
13. [Output Files](#output-files)
14. [Quick Evaluation](#quick-evaluation)

---

## Introduction

### What is Model Training?

Training a machine learning model means teaching it to recognize patterns in data. For NER, we show the model thousands of examples like:

```
Input:  "AC bus from Bangalore to Mumbai"
Output: [BUS_TYPE, O, O, SRC, O, DEST]
```

The model learns to predict the correct label for each word.

### What is Fine-Tuning?

We don't train from scratch. Instead, we **fine-tune** a pre-trained model (MiniLM). This means:

| Approach | Description | Time | Data Needed |
|----------|-------------|------|-------------|
| **From Scratch** | Train all weights randomly | Days/weeks | Millions of samples |
| **Fine-tuning** | Start with pre-trained weights | Minutes/hours | Hundreds/thousands |

MiniLM was pre-trained on billions of words, so it already "understands" language. We just teach it our specific task (bus NER).

### The 5-Step Pipeline

```
Step 1: Load BIO data from JSON
    ↓
Step 2: Split into train/eval datasets
    ↓
Step 3: Tokenize and align labels with subwords
    ↓
Step 4: Create model (MiniLM + classification head)
    ↓
Step 5: Train with HuggingFace Trainer
    ↓
Step 6: Save model and tokenizer
```

---

## Training Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  training_data_bio.json
          │
          ▼
  ┌───────────────────┐
  │   load_bio_data() │  ──→  List of {tokens, ner_tags}
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │ prepare_dataset() │  ──→  HuggingFace Dataset (train + eval split)
  └───────────────────┘
          │
          ▼
  ┌─────────────────────────────┐
  │ tokenize_and_align_labels() │  ──→  {input_ids, attention_mask, labels}
  └─────────────────────────────┘
          │
          ▼
  ┌───────────────────┐
  │   create_model()  │  ──→  MiniLM + Token Classification Head
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │   train_model()   │  ──→  Trained model (best F1 checkpoint)
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │   save_model()    │  ──→  model.safetensors, tokenizer, configs
  └───────────────────┘
```

---

## Step 1: Load BIO Data

### Function: `load_bio_data(data_path)`

**What it does:** Reads the BIO-tagged training data from a JSON file.

**Input:** Path to `training_data_bio.json`

**Output:** List of dictionaries, each with `tokens` and `ner_tags`

### Code

```python
def load_bio_data(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training samples from {data_path}")
    return data
```

### Example Data After This Step

```python
[
    {
        "tokens": ["VRL", "bus", "from", "Bangalore", "to", "delhi"],
        "ner_tags": [13, 0, 0, 1, 0, 3]
    },
    {
        "tokens": ["AC", "sleeper", "from", "Chennai", "to", "Mumbai"],
        "ner_tags": [5, 7, 0, 1, 0, 3]
    },
    # ... 2000 samples total
]
```

Where the numbers mean:
- 0 = O (Outside)
- 1 = B-SRC
- 3 = B-DEST
- 5 = B-BUS_TYPE
- 7 = B-SEAT_TYPE
- 13 = B-OPERATOR

---

## Step 2: Prepare Dataset

### Function: `prepare_dataset(data, test_split=0.1)`

**What it does:** 
1. Converts the list of dictionaries to a HuggingFace Dataset
2. Splits into training (90%) and evaluation (10%) sets

**Input:** List of samples from Step 1

**Output:** Two HuggingFace Dataset objects (train, eval)

### Code

```python
def prepare_dataset(data, test_split=0.1):
    # Convert to HF Dataset format
    dataset_dict = {
        "tokens": [sample["tokens"] for sample in data],
        "ner_tags": [sample["ner_tags"] for sample in data],
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train and eval
    split = dataset.train_test_split(test_size=test_split, seed=42)
    
    return split["train"], split["test"]
```

### Example Data After This Step

```
Train Dataset: 1800 samples
├── tokens: [["VRL", "bus", ...], ["AC", "sleeper", ...], ...]
└── ner_tags: [[13, 0, 0, 1, 0, 3], [5, 7, 0, 1, 0, 3], ...]

Eval Dataset: 200 samples
├── tokens: [["bus", "from", "Delhi", ...], ...]
└── ner_tags: [[0, 0, 1, ...], ...]
```

### Why Split?

| Dataset | Purpose | Size |
|---------|---------|------|
| **Train** | Model learns from these | 90% (1800) |
| **Eval** | Check if model is learning correctly | 10% (200) |

The eval set is **never** used for training - it's only for measuring performance.

---

## Step 3: Tokenize and Align Labels

### Function: `tokenize_and_align_labels(examples, tokenizer, label2id, max_length=128)`

**What it does:** This is the most critical function. It:
1. Converts word tokens to subword tokens (WordPiece)
2. Aligns the BIO labels with the subwords
3. Adds special tokens ([CLS], [SEP], [PAD])
4. Creates attention masks

**Input:** Batch of samples with `tokens` and `ner_tags`

**Output:** Tokenized inputs ready for the model

### The Subword Problem

MiniLM uses WordPiece tokenization, which may split words:

```
Word:      "Bangalore"
Subwords:  ["ban", "##galore"]  (2 tokens!)

Word:      "KSRTC"  
Subwords:  ["ks", "##rt", "##c"]  (3 tokens!)
```

**Problem:** We have ONE label for "Bangalore" (B-SRC), but now we have TWO tokens!

**Solution:** Only the FIRST subword gets the label. The rest get `-100` (ignored).

### Code Walkthrough

```python
def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=128):
    # Step 1: Tokenize (is_split_into_words=True tells it our input is pre-tokenized)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True,
    )
    
    labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        # Get which word each subword belongs to
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        
        label_sequence = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD]) → -100
                label_sequence.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word → actual label
                label_sequence.append(label_ids[word_idx])
            else:
                # Continuation subword → -100 (ignored)
                label_sequence.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_sequence)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```

### Detailed Example

**Input:**
```python
tokens = ["AC", "bus", "from", "Bangalore", "to", "Mumbai"]
ner_tags = [5, 0, 0, 1, 0, 3]
```

**After WordPiece tokenization:**
```
Position:  0      1     2     3      4     5       6        7      8     9      ...
Subword:  [CLS]   ac   bus   from   ban  ##gal   ##ore    to    mumbai [SEP]  [PAD]...
Word_id:   None    0     1     2      3     3       3       4      5    None   None...
```

**Label alignment:**
```
Position:    0      1    2    3    4     5      6     7    8      9     ...
Subword:   [CLS]   ac   bus  from ban  ##gal ##ore   to  mumbai [SEP] [PAD]...
Word_id:    None    0    1    2    3     3     3      4    5     None  None
Label:      -100    5    0    0    1   -100  -100     0    3    -100  -100...
```

**Explanation:**
- `[CLS]` has no word → `-100`
- `ac` is first token of word 0 → gets label `5` (B-BUS_TYPE)
- `ban` is first token of word 3 → gets label `1` (B-SRC)
- `##gal`, `##ore` are continuations → `-100`
- `[SEP]`, `[PAD]` are special → `-100`

### Why -100?

PyTorch's CrossEntropyLoss ignores positions with label `-100` when computing the loss. This means:
- The model is NOT penalized for predictions on subword continuations
- The model is NOT penalized for predictions on special tokens
- Only the FIRST subword of each word matters for training

---

## Step 4: Create Model

### Function: `create_model(model_name, num_labels, label2id, id2label)`

**What it does:** Creates a token classification model by:
1. Loading pre-trained MiniLM weights
2. Adding a classification head for our 19 labels

**Input:** 
- `model_name`: "microsoft/MiniLM-L12-H384-uncased"
- `num_labels`: 19 (O + 9 entities × 2 for B/I)
- Label mappings

**Output:** Ready-to-train model

### Code

```python
def create_model(model_name, num_labels, label2id, id2label):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    return model
```

### What Happens Internally

```
┌─────────────────────────────────────────────────────────────┐
│                    MiniLM Model                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: [CLS] ac bus from ban ##gal ##ore to mumbai [SEP]   │
│           ↓                                                  │
│  ┌─────────────────────────────────────┐                    │
│  │      Embedding Layer                 │                    │
│  │  Token Embeddings + Position Emb.    │                    │
│  └─────────────────────────────────────┘                    │
│           ↓                                                  │
│  ┌─────────────────────────────────────┐                    │
│  │      12 Transformer Layers           │  ← Pre-trained    │
│  │  (Self-Attention + Feed-Forward)     │    weights        │
│  └─────────────────────────────────────┘                    │
│           ↓                                                  │
│  Hidden states: [batch, seq_len, 384]                       │
│           ↓                                                  │
│  ┌─────────────────────────────────────┐                    │
│  │    Classification Head (NEW)         │  ← Randomly       │
│  │    Linear(384 → 19)                  │    initialized    │
│  └─────────────────────────────────────┘                    │
│           ↓                                                  │
│  Logits: [batch, seq_len, 19]                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** Only the classification head is randomly initialized. The 12 transformer layers already contain knowledge from pre-training.

---

## Step 5: Train Model

### Function: `train_model(...)`

**What it does:** Runs the training loop using HuggingFace's Trainer API.

### Training Configuration

```python
training_args = TrainingArguments(
    output_dir=output_dir,           # Where to save checkpoints
    num_train_epochs=10,             # Train for 10 epochs
    per_device_train_batch_size=16,  # 16 samples per batch
    per_device_eval_batch_size=16,
    learning_rate=5e-5,              # Small learning rate for fine-tuning
    weight_decay=0.01,               # L2 regularization
    eval_strategy="epoch",           # Evaluate after each epoch
    save_strategy="epoch",           # Save after each epoch
    load_best_model_at_end=True,     # Keep the best model
    metric_for_best_model="f1",      # Use F1 score to pick best
    greater_is_better=True,
    save_total_limit=2,              # Only keep 2 checkpoints
)
```

### What Happens During Training

```
Epoch 1/10
├── Batch 1: Forward pass → Loss = 2.34 → Backward pass → Update weights
├── Batch 2: Forward pass → Loss = 2.21 → Backward pass → Update weights
├── ...
├── Batch 113: Loss = 0.89
└── Evaluation: precision=0.75, recall=0.72, f1=0.73

Epoch 2/10
├── Batch 1: Loss = 0.67
├── ...
└── Evaluation: precision=0.85, recall=0.82, f1=0.83

...

Epoch 10/10
├── Batch 1: Loss = 0.08
├── ...
└── Evaluation: precision=0.96, recall=0.95, f1=0.95  ← Best model saved!
```

### The Training Loop (Simplified)

```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss  # Cross-entropy loss
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
    
    # Evaluate on eval set
    metrics = evaluate(model, eval_dataset)
    
    # Save if best so far
    if metrics["f1"] > best_f1:
        save_checkpoint(model)
```

---

## Step 6: Save Model

### Function: `save_model(trainer, tokenizer, output_path, label2id, id2label)`

**What it does:** Saves everything needed for inference:
- Model weights
- Tokenizer configuration
- Label mappings

### Code

```python
def save_model(trainer, tokenizer, output_path, label2id, id2label):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save label mappings
    with open(output_dir / "label2id.json", 'w') as f:
        json.dump(label2id, f, indent=2)
    
    with open(output_dir / "id2label.json", 'w') as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
```

---

## Understanding Subword Label Alignment

This is such an important concept that it deserves its own section.

### Why Does This Happen?

Transformer models use **subword tokenization** (WordPiece, BPE, etc.) because:
1. It handles unknown words by breaking them into known pieces
2. It keeps vocabulary size manageable (~30,000 tokens)
3. Rare words share components with common words

### Visual Example

```
Original:  "KSRTC bus from Bangalore to Mumbai tomorrow"
           ──────  ───  ────  ─────────  ──  ──────  ────────

Tokenized: [CLS] ks ##rt ##c bus from ban ##gal ##ore to mumbai tom ##orrow [SEP]
           ───── ── ──── ─── ─── ──── ─── ───── ───── ── ────── ─── ─────── ─────
Word ID:   None   0   0   0   1    2    3    3    3   4    5     6     6    None

Labels:    
Original:  [13,        0,    0,      1,         0,   3,       11           ]
           OPERATOR   O     O      SRC        O    DEST     DATE

Aligned:   [-100, 13, -100, -100, 0, 0, 1, -100, -100, 0, 3, 11, -100, -100]
                  ↑                    ↑              ↑  ↑  ↑
            First of KSRTC      First of Bangalore    │  │  First of tomorrow
                                                      │  B-DEST
                                                      O (to)
```

### The Rule

| Condition | Label Assigned |
|-----------|----------------|
| Special token ([CLS], [SEP], [PAD]) | -100 |
| First subword of a word | Original word's label |
| Continuation subword (##xxx) | -100 |

### Why -100 Specifically?

PyTorch's `CrossEntropyLoss` has a parameter `ignore_index=-100` by default. Any position with label -100 is:
- Excluded from loss computation
- Not counted in accuracy
- Effectively "invisible" to training

---

## Training Configuration Explained

### Key Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `num_train_epochs` | 10 | Enough iterations to converge |
| `learning_rate` | 5e-5 | Standard for fine-tuning transformers |
| `batch_size` | 16 | Fits in GPU memory, good gradient estimates |
| `weight_decay` | 0.01 | Prevents overfitting |
| `max_length` | 128 | Bus queries are short, no need for 512 |

### Learning Rate

```
5e-5 = 0.00005
```

This is much smaller than typical neural network training (0.001 or 0.01) because:
- Pre-trained weights are already good
- We want to adjust them slightly, not destroy them

### Evaluation Strategy

```python
eval_strategy="epoch"      # Run eval after each epoch
save_strategy="epoch"      # Save checkpoint after each epoch
load_best_model_at_end=True  # At the end, load the best checkpoint
metric_for_best_model="f1"   # "Best" = highest F1 score
```

This means if epoch 7 had the best F1, the final saved model is from epoch 7, not epoch 10.

---

## Metrics Computation

### Function: `compute_metrics(eval_pred, id2label)`

**What it does:** Calculates precision, recall, and F1 score using the seqeval library.

### Code

```python
def compute_metrics(eval_pred, id2label):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)  # Convert logits to predictions
    
    # Convert IDs to label strings, ignoring -100
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_labels = []
        
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:  # Only evaluate on real labels
                true_seq.append(id2label[label_id])
                pred_seq_labels.append(id2label[pred_id])
        
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_labels)
    
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }
```

### Understanding the Metrics

| Metric | Question Answered | Formula |
|--------|-------------------|---------|
| **Precision** | Of entities predicted, how many are correct? | TP / (TP + FP) |
| **Recall** | Of actual entities, how many did we find? | TP / (TP + FN) |
| **F1** | Balance of precision and recall | 2 × (P × R) / (P + R) |

### Example

```
True:  [B-SRC, O, O, B-DEST, O, B-DATE]
Pred:  [B-SRC, O, O, B-DEST, O, O     ]  ← Missed the DATE entity

Entities:
  True: SRC, DEST, DATE (3 entities)
  Pred: SRC, DEST (2 entities)

TP = 2 (SRC and DEST correct)
FP = 0 (no wrong predictions)
FN = 1 (missed DATE)

Precision = 2 / (2 + 0) = 1.0
Recall = 2 / (2 + 1) = 0.67
F1 = 2 × (1.0 × 0.67) / (1.0 + 0.67) = 0.80
```

---

## Model Architecture

### MiniLM-L12-H384

```
┌─────────────────────────────────────────────────────────────┐
│               microsoft/MiniLM-L12-H384-uncased             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Vocabulary Size: 30,522 tokens                             │
│  Hidden Size: 384                                           │
│  Layers: 12                                                 │
│  Attention Heads: 12                                        │
│  Parameters: ~22 million                                    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Input: Token IDs [batch, seq_len]                  │     │
│  └────────────────────────────────────────────────────┘     │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Embeddings [batch, seq_len, 384]                   │     │
│  │ (Token + Position + Token Type)                    │     │
│  └────────────────────────────────────────────────────┘     │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Transformer Layer 1                                 │     │
│  │ ├─ Multi-Head Self-Attention (12 heads)            │     │
│  │ ├─ Add & LayerNorm                                 │     │
│  │ ├─ Feed-Forward (384 → 1536 → 384)                 │     │
│  │ └─ Add & LayerNorm                                 │     │
│  └────────────────────────────────────────────────────┘     │
│                           ↓                                  │
│                         ...×12                               │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Hidden States [batch, seq_len, 384]                │     │
│  └────────────────────────────────────────────────────┘     │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Classification Head                                 │     │
│  │ Linear(384 → 19) + Dropout                         │     │
│  └────────────────────────────────────────────────────┘     │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Logits [batch, seq_len, 19]                        │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Output Interpretation 

For each token, the model outputs 19 numbers (logits). The highest one is the prediction:

```
Token: "Bangalore"
Logits: [0.1, 8.5, 0.2, 1.1, 0.3, ...]
         O   B-SRC I-SRC B-DEST ...
              ↑
         Highest = B-SRC → Prediction: B-SRC
```

---

## Output Files

After training, these files are saved to `models/bus_ner_transformer/`:

| File | Contents |
|------|----------|
| `model.safetensors` | Model weights (50-60 MB) |
| `config.json` | Model configuration (hidden size, num layers, etc.) |
| `tokenizer.json` | Tokenizer vocabulary and settings |
| `tokenizer_config.json` | Tokenizer configuration |
| `special_tokens_map.json` | Special token definitions |
| `vocab.txt` | Vocabulary list |
| `label2id.json` | Label to ID mapping |
| `id2label.json` | ID to label mapping |
| `training_args.bin` | Training arguments (for reproducibility) |

---

## Quick Evaluation

### Function: `quick_evaluation(model, tokenizer, id2label, test_queries)`

After training, we test on sample queries to verify the model works:

```python
test_queries = [
    "AC sleeper bus from Bangalore to Mumbai tomorrow",
    "show VRL buses from Chennai to Hyderabad tonight",
    "KSRTC Non-AC seater from Mysore to Mangalore",
]
```

### Example Output

```
Query: "AC sleeper bus from Bangalore to Mumbai tomorrow"
  - B-BUS_TYPE: "AC"
  - B-SEAT_TYPE: "sleeper"
  - B-SRC: "Bangalore"
  - B-DEST: "Mumbai"
  - B-DATE: "tomorrow"

Query: "show VRL buses from Chennai to Hyderabad tonight"
  - B-OPERATOR: "VRL"
  - B-SRC: "Chennai"
  - B-DEST: "Hyderabad"
  - B-TIME: "tonight"
```

---

## Summary

The training pipeline follows these steps:

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 1 | `load_bio_data()` | JSON file path | List of samples |
| 2 | `prepare_dataset()` | List of samples | Train/Eval Datasets |
| 3 | `tokenize_and_align_labels()` | Dataset | Tokenized with aligned labels |
| 4 | `create_model()` | Model name, labels | Model with classification head |
| 5 | `train_model()` | Model, datasets | Trained model |
| 6 | `save_model()` | Trained model | Saved files |

**Key takeaways:**
1. We **fine-tune** a pre-trained model (not train from scratch)
2. **Subword alignment** is critical - only first subword gets the label
3. **-100** is used to ignore tokens during loss computation
4. **F1 score** is used to select the best model
5. The Trainer API handles the training loop automatically

The resulting model can extract entities from bus queries with ~95%+ F1 score.