# Data Generation Process - Step by Step Guide

This document explains in detail how the training data for the Bus NER model is generated using `src/data_generator.py`. We'll walk through each function, show exactly what happens at each step, and demonstrate how data is transformed from templates to the final BIO-tagged format.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Data Flow Overview](#data-flow-overview)
3. [Building Blocks: Entity Pools and Templates](#building-blocks-entity-pools-and-templates)
4. [Step-by-Step Function Explanations](#step-by-step-function-explanations)
5. [Character-to-Token Mapping](#character-to-token-mapping)
6. [BIO Tag Assignment](#bio-tag-assignment)
7. [Complete Label Mapping](#complete-label-mapping)
8. [Full Walkthrough Examples](#full-walkthrough-examples)
9. [Output Files](#output-files)

---

## Introduction

### Why Synthetic Data Generation?

Training an NER model requires labeled data where each word/token has a tag. There are two ways to get this data:

| Approach | Pros | Cons |
|----------|------|------|
| **Manual Annotation** | High quality, diverse | Expensive, slow, error-prone |
| **Template Generation** | Fast, scalable, error-free | Limited to template patterns |

We use **template-based generation** because:
1. **No annotation errors** - labels are computed automatically
2. **Scalable** - can generate thousands of samples in seconds
3. **Controllable** - easy to add new patterns or entity values
4. **Perfect for domain-specific NER** - we know exactly what patterns users will type

### What Does the Generator Produce?

The generator creates two output files:

| File | Format | Purpose |
|------|--------|---------|
| `training_data.json` | Span format | Human-readable, character positions |
| `training_data_bio.json` | BIO format | Model training input |

---

## Data Flow Overview

Here's how data flows through the generation pipeline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA GENERATION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

     ENTITY_VALUES                    TEMPLATES
     ─────────────                    ─────────
     SRC: [Bangalore, Mumbai...]      "bus from {SRC} to {DEST}"
     DEST: [Chennai, Delhi...]        "{BUS_TYPE} bus from {SRC} to {DEST}"
     BUS_TYPE: [AC, Non-AC...]        ...100+ templates
              │                                │
              └────────────┬───────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    fill_template()     │
              │  Pick random values    │
              │  Replace placeholders  │
              │  Calculate positions   │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    SPAN FORMAT         │
              │ {                      │
              │   "text": "AC bus...", │
              │   "entities": [        │
              │     [0, 2, "BUS_TYPE"] │
              │   ]                    │
              │ }                      │
              └────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
   training_data.json              convert_span_to_bio()
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │    BIO FORMAT          │
                               │ {                      │
                               │   "tokens": ["AC",...],│
                               │   "ner_tags": [5, 0...]│
                               │ }                      │
                               └────────────────────────┘
                                           │
                                           ▼
                               training_data_bio.json
```

---

## Building Blocks: Entity Pools and Templates

### Entity Value Pools (ENTITY_VALUES)

The generator has predefined lists of values for each entity type:

```python
ENTITY_VALUES = {
    "SRC": [
        "Bangalore", "Mumbai", "Chennai", "Hyderabad", "Delhi",
        "Pune", "Kolkata", "bangalore", "mumbai", ...  # 30 values
    ],
    "DEST": [
        "Bangalore", "Mumbai", "Chennai", ...  # Same cities
    ],
    "BUS_TYPE": [
        "AC", "Non-AC", "ac", "non-ac"  # 4 values
    ],
    "SEAT_TYPE": [
        "Sleeper", "Seater", "sleeper", "seater", "semi sleeper"  # 5 values
    ],
    "TIME": [
        "morning", "evening", "6 am", "10 pm", "early morning", ...  # 14 values
    ],
    "DATE": [
        "today", "tomorrow", "Monday", "next Friday", "15th January", ...  # 20 values
    ],
    "OPERATOR": [
        "KSRTC", "VRL", "RedBus", "Orange", ...  # 14 values
    ],
    "BOARDING_POINT": [
        "Majestic", "Silk Board", "Electronic City", ...  # 15 values
    ],
    "DROPPING_POINT": [
        "KPHB", "Hitec City", "Koyambedu", ...  # 20 values
    ]
}
```

**Key Design Decisions:**
- Both capitalized and lowercase variants are included (helps model generalize)
- Multi-word entities like "Electronic City" and "semi sleeper" are included
- SRC and DEST use the same city pool (but the generator ensures they're different)

### Query Templates (TEMPLATES)

Templates define query patterns with `{ENTITY_TYPE}` placeholders:

```python
TEMPLATES = [
    # Basic queries
    "bus from {SRC} to {DEST}",
    "{SRC} to {DEST} bus",
    
    # With bus type
    "{BUS_TYPE} bus from {SRC} to {DEST}",
    
    # With multiple entities
    "{OPERATOR} {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE} {TIME}",
    
    # Natural language
    "I need a bus from {SRC} to {DEST}",
    "can you find bus from {SRC} to {DEST} {DATE}",
    
    # ... 100+ templates total
]
```

**Why so many templates?**
- Different word orders: "from X to Y" vs "X to Y"
- Different entity combinations: just SRC/DEST, or all 9 entities
- Natural language variations: "show", "find", "book", "I need"
- Questions: "is there a bus", "any buses available"

---

## Step-by-Step Function Explanations

### Function 1: `get_random_value(entity_type, exclude)`

**What it does:** Picks a random value from the entity pool.

**Input:**
- `entity_type`: String like "SRC", "DEST", "BUS_TYPE"
- `exclude`: Optional list of values to NOT pick (used to ensure SRC ≠ DEST)

**Output:** A random string value

**Example:**
```python
get_random_value("BUS_TYPE")
# Returns: "AC" (or "Non-AC", "ac", "non-ac")

get_random_value("DEST", exclude=["Mumbai"])
# Returns: Any city EXCEPT "Mumbai"
```

**Code:**
```python
def get_random_value(entity_type: str, exclude: List[str] = None) -> str:
    values = ENTITY_VALUES[entity_type]
    if exclude:
        values = [v for v in values if v not in exclude]
    return random.choice(values)
```

---

### Function 2: `fill_template(template)`

**What it does:** Takes a template, fills in random values, and calculates where each entity is located.

**Input:** Template string like `"bus from {SRC} to {DEST}"`

**Output:** Tuple of (filled_text, entity_spans)

**Step-by-step example:**

```
INPUT:  "bus from {SRC} to {DEST}"

STEP 1: Find placeholders
        → [("SRC", position 9), ("DEST", position 18)]

STEP 2: Pick random values (ensuring SRC ≠ DEST)
        → SRC = "Bangalore", DEST = "Mumbai"

STEP 3: Replace placeholders one by one
        "bus from {SRC} to {DEST}"
        → "bus from Bangalore to {DEST}"
        → "bus from Bangalore to Mumbai"

STEP 4: Calculate character positions
        "bus from Bangalore to Mumbai"
         0123456789...
        
        "Bangalore" starts at position 9, ends at 18
        "Mumbai" starts at position 22, ends at 28

OUTPUT: ("bus from Bangalore to Mumbai", 
         [(9, 18, "SRC"), (22, 28, "DEST")])
```

**Code walkthrough:**
```python
def fill_template(template: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    # Step 1: Find all {ENTITY} placeholders
    pattern = r'\{([A-Z_]+)\}'
    placeholders = [(m.group(1), m.start()) for m in re.finditer(pattern, template)]
    
    # Step 2: Pick values for each placeholder
    used_values = {}
    for entity_type, template_pos in placeholders:
        # Ensure SRC ≠ DEST
        exclude = []
        if entity_type == "DEST" and "SRC" in used_values:
            exclude = [used_values["SRC"]]
        
        value = get_random_value(entity_type, exclude)
        used_values[entity_type] = value
        
        # Replace in template
        result = result.replace("{" + entity_type + "}", value, 1)
    
    # Step 3: Find actual character spans
    spans = find_entity_spans(result, entity_positions)
    
    return result, spans
```

---

### Function 3: `find_entity_spans(text, entity_positions)`

**What it does:** Finds the exact character start/end positions of each entity in the filled text.

**Why is this needed?** After replacing placeholders, the positions shift. "Bangalore" is 9 characters but "{SRC}" was only 5.

**Input:**
- `text`: The filled text
- `entity_positions`: List of (entity_type, value, approximate_position)

**Output:** List of (start_char, end_char, entity_type)

**Example:**
```python
text = "bus from Bangalore to Mumbai"
entity_positions = [("SRC", "Bangalore", 9), ("DEST", "Mumbai", 22)]

# The function searches for each value in order
# "Bangalore" found at index 9
# "Mumbai" found at index 22 (searching from index 18 onwards)

OUTPUT: [(9, 18, "SRC"), (22, 28, "DEST")]
```

**Code:**
```python
def find_entity_spans(text, entity_positions):
    spans = []
    search_start = 0
    
    # Sort by position to process in order
    sorted_entities = sorted(entity_positions, key=lambda x: x[2])
    
    for entity_type, entity_value, _ in sorted_entities:
        # Find the value starting from where we left off
        start_idx = text.find(entity_value, search_start)
        if start_idx != -1:
            end_idx = start_idx + len(entity_value)
            spans.append((start_idx, end_idx, entity_type))
            search_start = end_idx  # Continue searching after this entity
    
    return spans
```

---

### Function 4: `generate_training_sample()`

**What it does:** Creates one complete training sample in span format.

**Process:**
1. Pick a random template
2. Fill it with random values
3. Return the text and entity spans

**Output format:**
```python
{
    "text": "AC bus from Bangalore to Mumbai",
    "entities": [
        [0, 2, "BUS_TYPE"],      # "AC" at positions 0-2
        [12, 21, "SRC"],         # "Bangalore" at 12-21
        [25, 31, "DEST"]         # "Mumbai" at 25-31
    ]
}
```

---

### Function 5: `generate_training_data(num_samples, seed)`

**What it does:** Generates multiple samples by calling `generate_training_sample()` repeatedly.

**Input:**
- `num_samples`: How many samples to generate (default: 2000)
- `seed`: Random seed for reproducibility

**Output:** List of span-format samples

---

### Function 6: `convert_span_to_bio(sample)`

**What it does:** Converts span-format data to BIO-tagged format for transformer training.

This is the most complex function. Let's break it down:

**Input:**
```python
{
    "text": "bus from Bangalore to Mumbai",
    "entities": [[9, 18, "SRC"], [22, 28, "DEST"]]
}
```

**Output:**
```python
{
    "tokens": ["bus", "from", "Bangalore", "to", "Mumbai"],
    "ner_tags": [0, 0, 1, 0, 3]
}
```

Where the numbers are label IDs:
- 0 = "O" (Outside any entity)
- 1 = "B-SRC" (Beginning of SRC entity)
- 3 = "B-DEST" (Beginning of DEST entity)

**Detailed walkthrough in next section.**

---

## Character-to-Token Mapping

The key challenge: entities are defined by **character positions**, but transformers work with **tokens**. We need to map characters to tokens.

### Step-by-Step Mapping Process

**Input text:** `"bus from Bangalore to Mumbai"`

**Step 1: Split into tokens (whitespace)**
```
tokens = ["bus", "from", "Bangalore", "to", "Mumbai"]
index:      0      1         2          3       4
```

**Step 2: Build character → token mapping**

We track which character index belongs to which token:

```
Character: b  u  s     f  r  o  m     B  a  n  g  a  l  o  r  e     t  o     M  u  m  b  a  i
Index:     0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
Token:     0  0  0  -  1  1  1  1  -  2  2  2  2  2  2  2  2  2  -  3  3  -  4  4  4  4  4  4
```

Note: Spaces (index 3, 8, 18, 21) don't belong to any token (marked as `-`).

**Code that builds this mapping:**
```python
char_to_token = {}
current_char = 0

for token_idx, token in enumerate(tokens):
    for i in range(len(token)):
        char_to_token[current_char + i] = token_idx
    current_char += len(token) + 1  # +1 for space
```

**Result:**
```python
char_to_token = {
    0: 0, 1: 0, 2: 0,           # "bus" → token 0
    4: 1, 5: 1, 6: 1, 7: 1,     # "from" → token 1
    9: 2, 10: 2, ..., 17: 2,    # "Bangalore" → token 2
    19: 3, 20: 3,               # "to" → token 3
    22: 4, ..., 27: 4           # "Mumbai" → token 4
}
```

---

## BIO Tag Assignment

### How Tags Are Assigned

Once we know which characters map to which tokens, we assign BIO tags:

**Entity span:** `[9, 18, "SRC"]` means characters 9-17 (exclusive end) are SRC.

**Step 1: Find which tokens contain these characters**
```python
for char_idx in range(9, 18):  # Characters 9, 10, 11, ..., 17
    token_idx = char_to_token[char_idx]  # All map to token 2
```

→ Token 2 ("Bangalore") contains the SRC entity

**Step 2: Assign B- or I- tags**
- First token of entity → `B-SRC`
- Additional tokens → `I-SRC`

Since "Bangalore" is a single token, it just gets `B-SRC`.

### Multi-Word Entity Example

**Text:** `"pickup at Electronic City"`

**Tokens:** `["pickup", "at", "Electronic", "City"]`

**Entity:** `[10, 26, "BOARDING_POINT"]` (characters 10-25 = "Electronic City")

**Character mapping:**
```
pickup  at  Electronic  City
0-5     7-8  10-19      21-24

char_to_token:
10-19 → token 2 ("Electronic")
21-24 → token 3 ("City")
```

**Tag assignment:**
```python
tokens:   ["pickup", "at", "Electronic",          "City"]
ner_tags: [   0,      0,   B-BOARDING_POINT, I-BOARDING_POINT]
          [   0,      0,        15,                  16        ]
```

The first token gets `B-` (Beginning), the second gets `I-` (Inside).

---

## Complete Label Mapping

Here's the complete mapping from `label2id.json`:

| Label | ID | Meaning |
|-------|-----|---------|
| O | 0 | Outside (not an entity) |
| B-SRC | 1 | Beginning of Source city |
| I-SRC | 2 | Inside Source city (continuation) |
| B-DEST | 3 | Beginning of Destination city |
| I-DEST | 4 | Inside Destination city |
| B-BUS_TYPE | 5 | Beginning of Bus type |
| I-BUS_TYPE | 6 | Inside Bus type |
| B-SEAT_TYPE | 7 | Beginning of Seat type |
| I-SEAT_TYPE | 8 | Inside Seat type |
| B-TIME | 9 | Beginning of Time |
| I-TIME | 10 | Inside Time |
| B-DATE | 11 | Beginning of Date |
| I-DATE | 12 | Inside Date |
| B-OPERATOR | 13 | Beginning of Operator |
| I-OPERATOR | 14 | Inside Operator |
| B-BOARDING_POINT | 15 | Beginning of Boarding point |
| I-BOARDING_POINT | 16 | Inside Boarding point |
| B-DROPPING_POINT | 17 | Beginning of Dropping point |
| I-DROPPING_POINT | 18 | Inside Dropping point |

**Total: 19 labels** (1 for O + 9 entities × 2 for B/I)

---

## Full Walkthrough Examples

### Example 1: Simple Query

**Template:** `"bus from {SRC} to {DEST}"`

**Step 1: Fill template**
```
Random values: SRC = "Bangalore", DEST = "Mumbai"
Filled text: "bus from Bangalore to Mumbai"
```

**Step 2: Calculate spans**
```
Text:    "bus from Bangalore to Mumbai"
Index:    0123456789...

"Bangalore" → start=9, end=18
"Mumbai"    → start=22, end=28

Spans: [[9, 18, "SRC"], [22, 28, "DEST"]]
```

**Step 3: Span format output**
```json
{
  "text": "bus from Bangalore to Mumbai",
  "entities": [[9, 18, "SRC"], [22, 28, "DEST"]]
}
```

**Step 4: Convert to BIO**
```
Tokens:    ["bus", "from", "Bangalore", "to", "Mumbai"]
Token idx:    0      1          2         3       4

Character ranges:
  Token 0: chars 0-2
  Token 1: chars 4-7
  Token 2: chars 9-17   ← contains SRC (9-18)
  Token 3: chars 19-20
  Token 4: chars 22-27  ← contains DEST (22-28)

BIO tags:  [  O,     O,     B-SRC,    O,   B-DEST]
Tag IDs:   [  0,     0,       1,      0,      3   ]
```

**Step 5: BIO format output**
```json
{
  "tokens": ["bus", "from", "Bangalore", "to", "Mumbai"],
  "ner_tags": [0, 0, 1, 0, 3]
}
```

---

### Example 2: Multi-Word Entities

**Template:** `"{BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST}"`

**Filled text:** `"AC semi sleeper from Chennai to Delhi"`

**Spans:**
```
"AC"          → [0, 2, "BUS_TYPE"]
"semi sleeper"→ [3, 15, "SEAT_TYPE"]  (multi-word!)
"Chennai"     → [21, 28, "SRC"]
"Delhi"       → [32, 37, "DEST"]
```

**Tokens:** `["AC", "semi", "sleeper", "from", "Chennai", "to", "Delhi"]`

**Character-to-token mapping:**
```
Token 0 "AC":      chars 0-1
Token 1 "semi":    chars 3-6
Token 2 "sleeper": chars 8-14
Token 3 "from":    chars 16-19
Token 4 "Chennai": chars 21-27
Token 5 "to":      chars 29-30
Token 6 "Delhi":   chars 32-36
```

**Entity → Token mapping:**
```
BUS_TYPE [0-2]   → Token 0         → B-BUS_TYPE
SEAT_TYPE [3-15] → Tokens 1, 2     → B-SEAT_TYPE, I-SEAT_TYPE
SRC [21-28]      → Token 4         → B-SRC
DEST [32-37]     → Token 6         → B-DEST
```

**Final BIO output:**
```json
{
  "tokens": ["AC", "semi", "sleeper", "from", "Chennai", "to", "Delhi"],
  "ner_tags": [5, 7, 8, 0, 1, 0, 3]
}
```

Decoded:
| Token | Tag ID | Tag Label |
|-------|--------|-----------|
| AC | 5 | B-BUS_TYPE |
| semi | 7 | B-SEAT_TYPE |
| sleeper | 8 | I-SEAT_TYPE |
| from | 0 | O |
| Chennai | 1 | B-SRC |
| to | 0 | O |
| Delhi | 3 | B-DEST |

---

### Example 3: Complex Query with Many Entities

**Template:** `"{OPERATOR} {BUS_TYPE} {SEAT_TYPE} from {SRC} to {DEST} {DATE} {TIME} pickup at {BOARDING_POINT} drop at {DROPPING_POINT}"`


**Filled text:** 
```
"VRL AC Sleeper from Bangalore to Mumbai tomorrow morning pickup at Electronic City drop at Hitec City"
```

**Spans:**
| Entity | Start | End | Text |
|--------|-------|-----|------|
| OPERATOR | 0 | 3 | VRL |
| BUS_TYPE | 4 | 6 | AC |
| SEAT_TYPE | 7 | 14 | Sleeper |
| SRC | 20 | 29 | Bangalore |
| DEST | 33 | 39 | Mumbai |
| DATE | 40 | 48 | tomorrow |
| TIME | 49 | 56 | morning |
| BOARDING_POINT | 67 | 82 | Electronic City |
| DROPPING_POINT | 91 | 101 | Hitec City |

**Tokens (14 total):**
```
["VRL", "AC", "Sleeper", "from", "Bangalore", "to", "Mumbai", 
 "tomorrow", "morning", "pickup", "at", "Electronic", "City", 
 "drop", "at", "Hitec", "City"]
```

**BIO tags:**
```
[B-OPERATOR, B-BUS_TYPE, B-SEAT_TYPE, O, B-SRC, O, B-DEST,
 B-DATE, B-TIME, O, O, B-BOARDING_POINT, I-BOARDING_POINT,
 O, O, B-DROPPING_POINT, I-DROPPING_POINT]

[13, 5, 7, 0, 1, 0, 3, 11, 9, 0, 0, 15, 16, 0, 0, 17, 18]
```

---

## Output Files

### training_data.json (Span Format)

This file contains the human-readable format with character positions:

```json
[
  {
    "text": "VRL bus from Bangalore to delhi Saturday night...",
    "entities": [
      [0, 3, "OPERATOR"],
      [13, 22, "SRC"],
      [26, 31, "DEST"],
      ...
    ]
  },
  ...
]
```

### training_data_bio.json (BIO Format)

This file is what the transformer model actually trains on:

```json
[
  {"tokens": ["VRL", "bus", "from", "Bangalore", "to", "delhi", ...], "ner_tags": [13, 0, 0, 1, 0, 3, ...]},
  {"tokens": ["book", "a", "bus", "from", "Hyderabad", ...], "ner_tags": [0, 0, 0, 0, 1, ...]},
  ...
]
```

### label2id.json & id2label.json

These files store the mapping between label names and numeric IDs:

```json
// label2id.json
{
  "O": 0,
  "B-SRC": 1,
  "I-SRC": 2,
  ...
}

// id2label.json  
{
  "0": "O",
  "1": "B-SRC",
  "2": "I-SRC",
  ...
}
```

---

## Summary

The data generation process follows these steps:

1. **Pick a template** randomly from 100+ patterns
2. **Fill placeholders** with random entity values (ensuring SRC ≠ DEST)
3. **Calculate character spans** for each entity
4. **Save span format** to `training_data.json`
5. **Convert to BIO format**:
   - Tokenize text by whitespace
   - Map characters to tokens
   - Assign B- tag to first token of entity, I- to rest
   - Convert tag names to numeric IDs
6. **Save BIO format** to `training_data_bio.json`

The result is perfectly labeled training data ready for transformer fine-tuning, with zero manual annotation effort.
