
import json
import random
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from constants.entity_values import ENTITY_VALUES
from constants.templates import TEMPLATES
from constants.entity_labels import ENTITY_LABELS




def get_random_value(entity_type: str, exclude: List[str] = None) -> str:
    """
    Get a random value for an entity type.
    
    Args:
        entity_type: The type of entity (SRC, DEST, etc.)
        exclude: Values to exclude (e.g., SRC value when picking DEST)
    
    Returns:
        A random value for the entity type
    """
    values = ENTITY_VALUES[entity_type]
    if exclude:
        values = [v for v in values if v not in exclude]
    return random.choice(values)


def find_entity_spans(text: str, entity_positions: List[Tuple[str, str, int]]) -> List[Tuple[int, int, str]]:
    """
    Convert entity positions to character spans.
    
    This function finds the exact character positions of entities in the text
    without manual index counting - it uses the insertion order and string search.
    
    Args:
        text: The complete query text
        entity_positions: List of (entity_type, entity_value, template_position)
    
    Returns:
        List of (start_char, end_char, entity_type) tuples
    """
    spans = []
    search_start = 0
    
    # Sort by template position to find entities in order
    sorted_entities = sorted(entity_positions, key=lambda x: x[2])
    
    for entity_type, entity_value, _ in sorted_entities:
        # Find the entity value in the text starting from search_start
        start_idx = text.find(entity_value, search_start)
        if start_idx != -1:
            end_idx = start_idx + len(entity_value)
            spans.append((start_idx, end_idx, entity_type))
            # Move search start past this entity to handle duplicates correctly
            search_start = end_idx
    
    return spans


def fill_template(template: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Fill a template with random entity values and compute spans.
    
    Args:
        template: Query template with {ENTITY_TYPE} placeholders
    
    Returns:
        Tuple of (filled_text, entity_spans)
    """
    # Find all placeholders in the template
    pattern = r'\{([A-Z_]+)\}'
    placeholders = [(m.group(1), m.start()) for m in re.finditer(pattern, template)]
    
    if not placeholders:
        return template, []
    
    # Track values to avoid same source/destination values
    used_values = {}
    entity_positions = []
    
    # Build the filled text by replacing placeholders
    result = template
    offset = 0  # Track offset as we replace text
    
    for entity_type, template_pos in placeholders:
        # Keep source/destination entities different for better training quality.
        exclude = []
        if entity_type == "DESTINATION_NAME" and "SOURCE_NAME" in used_values:
            exclude = [used_values["SOURCE_NAME"]]
        elif entity_type == "SOURCE_NAME" and "DESTINATION_NAME" in used_values:
            exclude = [used_values["DESTINATION_NAME"]]
        elif entity_type == "DESTINATION_CITY_CODE" and "SOURCE_CITY_CODE" in used_values:
            exclude = [used_values["SOURCE_CITY_CODE"]]
        elif entity_type == "SOURCE_CITY_CODE" and "DESTINATION_CITY_CODE" in used_values:
            exclude = [used_values["DESTINATION_CITY_CODE"]]
        
        value = get_random_value(entity_type, exclude)
        used_values[entity_type] = value
        
        # Calculate position in the result string
        placeholder = "{" + entity_type + "}"
        pos_in_result = result.find(placeholder)
        
        # Record the entity position before replacement
        entity_positions.append((entity_type, value, pos_in_result))
        
        # Replace the placeholder
        result = result.replace(placeholder, value, 1)
    
    # Now find the actual spans in the final text
    spans = find_entity_spans(result, entity_positions)
    
    return result, spans


def generate_training_sample() -> Dict[str, Any]:
    """
    Generate a single training sample.
    
    Returns:
        Dict with 'text' and 'entities' keys in spaCy format
    """
    template = random.choice(TEMPLATES)
    text, spans = fill_template(template)
    
    # Convert spans to spaCy format: [[start, end, label], ...]
    entities = [[start, end, label] for start, end, label in spans]
    
    return {
        "text": text,
        "entities": entities
    }


def generate_training_data(num_samples: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate multiple training samples.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of training samples in spaCy format
    """
    random.seed(seed)
    samples = []
    
    for _ in range(num_samples):
        sample = generate_training_sample()
        samples.append(sample)
    
    return samples


def validate_sample(sample: Dict[str, Any]) -> bool:
    """
    Validate that a sample's entity spans are correct.
    
    Args:
        sample: A training sample with 'text' and 'entities'
    
    Returns:
        True if all entity spans are valid
    """
    text = sample["text"]
    for start, end, label in sample["entities"]:
        # Check bounds
        if start < 0 or end > len(text) or start >= end:
            print(f"Invalid span: [{start}, {end}] for text of length {len(text)}")
            return False
        
        # Extract and verify the entity text exists
        entity_text = text[start:end]
        if not entity_text.strip():
            print(f"Empty entity text at [{start}, {end}]")
            return False
    
    return True


def save_training_data(samples: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save training data to a JSON file.
    
    Args:
        samples: List of training samples
        output_path: Path to save the JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(samples)} samples to {output_path}")


def print_sample_examples(samples: List[Dict[str, Any]], num_examples: int = 5) -> None:
    """
    Print sample examples with highlighted entities for verification.
    
    Args:
        samples: List of training samples
        num_examples: Number of examples to print
    """
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING DATA EXAMPLES")
    print("=" * 60)
    
    for i, sample in enumerate(samples[:num_examples]):
        text = sample["text"]
        entities = sample["entities"]
        
        print(f"\n[{i+1}] Text: \"{text}\"")
        print("    Entities:")
        
        for start, end, label in entities:
            entity_text = text[start:end]
            print(f"      - {label}: \"{entity_text}\" [{start}:{end}]")


# =============================================================================
# BIO TAG CONVERSION FOR TRANSFORMERS
# =============================================================================

# Build complete BIO label list
def get_bio_labels() -> List[str]:
    """
    Generate the complete list of BIO labels.
    
    Returns:
        List of labels: ["O", "B-SRC", "I-SRC", "B-DEST", ...]
    """
    labels = ["O"]
    for entity in ENTITY_LABELS:
        labels.append(f"B-{entity}")
        labels.append(f"I-{entity}")
    return labels


def get_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Get label to ID and ID to label mappings.
    
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    labels = get_bio_labels()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label


def convert_span_to_bio(
    sample: Dict[str, Any],
    tokenizer=None
) -> Optional[Dict[str, Any]]:
    """
    Convert a span-based annotation sample to BIO format.
    
    Uses simple whitespace tokenization for compatibility with transformer
    tokenizers that will handle subword tokenization during training.
    
    Args:
        sample: Dict with 'text' and 'entities' (span format)
        tokenizer: Optional tokenizer (not used, for future compatibility)
    
    Returns:
        Dict with 'tokens' and 'ner_tags' as label IDs, or None if invalid
    """
    text = sample["text"]
    entities = sample["entities"]  # [[start, end, label], ...]
    
    # Simple whitespace tokenization
    tokens = text.split()
    
    if not tokens:
        return None
    
    # Build character offset to token index mapping
    char_to_token = {}
    current_char = 0
    
    for token_idx, token in enumerate(tokens):
        for i in range(len(token)):
            char_to_token[current_char + i] = token_idx
        current_char += len(token) + 1  # +1 for space
    
    # Initialize all tags as "O"
    label2id, _ = get_label_mappings()
    ner_tags = [label2id["O"]] * len(tokens)
    
    # Assign BIO tags based on entity spans
    for start, end, label in entities:
        # Find tokens that overlap with this entity span
        entity_token_indices = set()
        for char_idx in range(start, end):
            if char_idx in char_to_token:
                entity_token_indices.add(char_to_token[char_idx])
        
        if not entity_token_indices:
            continue
        
        # Sort token indices and assign B/I tags
        sorted_indices = sorted(entity_token_indices)
        for i, token_idx in enumerate(sorted_indices):
            if i == 0:
                ner_tags[token_idx] = label2id[f"B-{label}"]
            else:
                ner_tags[token_idx] = label2id[f"I-{label}"]
    
    return {
        "tokens": tokens,
        "ner_tags": ner_tags
    }


def convert_samples_to_bio(
    samples: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert multiple span-based samples to BIO format.
    
    Args:
        samples: List of samples with 'text' and 'entities'
    
    Returns:
        List of samples with 'tokens' and 'ner_tags'
    """
    bio_samples = []
    skipped = 0
    
    for sample in samples:
        bio_sample = convert_span_to_bio(sample)
        if bio_sample:
            bio_samples.append(bio_sample)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"Skipped {skipped} samples during BIO conversion")
    
    return bio_samples


def save_bio_training_data(
    bio_samples: List[Dict[str, Any]], 
    output_path: str
) -> None:
    """
    Save BIO-tagged training data to a JSON file (compact horizontal format).
    
    Each sample is written on a single line for better readability.
    
    Args:
        bio_samples: List of samples with 'tokens' and 'ner_tags'
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("[\n")
        for i, sample in enumerate(bio_samples):
            line = json.dumps(sample, ensure_ascii=False)
            if i < len(bio_samples) - 1:
                f.write(f"  {line},\n")
            else:
                f.write(f"  {line}\n")
        f.write("]\n")
    
    print(f"Saved {len(bio_samples)} BIO-tagged samples to {output_path}")


def save_label_mappings(output_dir: str) -> None:
    """
    Save label mappings to JSON files for model loading.
    
    Args:
        output_dir: Directory to save the mapping files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    label2id, id2label = get_label_mappings()
    
    # Save label2id
    with open(os.path.join(output_dir, "label2id.json"), 'w') as f:
        json.dump(label2id, f, indent=2)
    
    # Save id2label (convert int keys to strings for JSON)
    id2label_str = {str(k): v for k, v in id2label.items()}
    with open(os.path.join(output_dir, "id2label.json"), 'w') as f:
        json.dump(id2label_str, f, indent=2)
    
    print(f"Saved label mappings to {output_dir}")


def print_bio_examples(bio_samples: List[Dict[str, Any]], num_examples: int = 3) -> None:
    """
    Print BIO-tagged examples for verification.
    
    Args:
        bio_samples: List of BIO-tagged samples
        num_examples: Number of examples to print
    """
    _, id2label = get_label_mappings()
    
    print("\n" + "=" * 60)
    print("BIO-TAGGED EXAMPLES")
    print("=" * 60)
    
    for i, sample in enumerate(bio_samples[:num_examples]):
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]
        
        print(f"\n[{i+1}] Tokens and Tags:")
        for token, tag_id in zip(tokens, ner_tags):
            tag = id2label[tag_id]
            print(f"    {token:20} -> {tag}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    NUM_SAMPLES = 2000  # Increased for better coverage
    BASE_DIR = Path(__file__).parent.parent
    SPAN_OUTPUT_PATH = BASE_DIR / "data" / "training_data.json"
    BIO_OUTPUT_PATH = BASE_DIR / "data" / "training_data_bio.json"
    LABEL_DIR = BASE_DIR / "data"
    
    print("Bus NER Data Generator")
    print("=" * 60)
    print(f"Generating {NUM_SAMPLES} training samples...")
    
    # Generate span-based data
    samples = generate_training_data(num_samples=NUM_SAMPLES)
    
    # Validate all samples
    print("\nValidating samples...")
    valid_count = sum(1 for s in samples if validate_sample(s))
    print(f"Valid samples: {valid_count}/{len(samples)}")
    
    # Print span-based examples
    print_sample_examples(samples)
    
    # Save span-based data (for reference)
    print("\n" + "=" * 60)
    save_training_data(samples, str(SPAN_OUTPUT_PATH))
    
    # Convert to BIO format
    print("\n" + "=" * 60)
    print("Converting to BIO format for transformer training...")
    bio_samples = convert_samples_to_bio(samples)
    
    # Print BIO examples
    print_bio_examples(bio_samples)
    
    # Save BIO data
    print("\n" + "=" * 60)
    save_bio_training_data(bio_samples, str(BIO_OUTPUT_PATH))
    
    # Save label mappings
    save_label_mappings(str(LABEL_DIR))
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"  Span format: {SPAN_OUTPUT_PATH}")
    print(f"  BIO format:  {BIO_OUTPUT_PATH}")
    print(f"  Labels:      {LABEL_DIR}/label2id.json, id2label.json")
    print(f"  Total labels: {len(get_bio_labels())}")

