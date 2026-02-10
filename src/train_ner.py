
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from constants.entity_labels import ENTITY_LABELS



# CONFIGURATION


MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"



# LABEL MANAGEMENT


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



# DATA LOADING


def load_bio_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load BIO-tagged training data from JSON file.
    
    Args:
        data_path: Path to the BIO training data JSON file
    
    Returns:
        List of samples with 'tokens' and 'ner_tags'
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training samples from {data_path}")
    return data


def prepare_dataset(data: List[Dict[str, Any]], test_split: float = 0.1) -> Tuple[Dataset, Dataset]:
    """
    Prepare Hugging Face datasets for training and evaluation.
    
    Args:
        data: List of BIO-tagged samples
        test_split: Fraction of data for testing
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Convert to HF Dataset format
    dataset_dict = {
        "tokens": [sample["tokens"] for sample in data],
        "ner_tags": [sample["ner_tags"] for sample in data],
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train and eval
    split = dataset.train_test_split(test_size=test_split, seed=42)
    
    print(f"Train samples: {len(split['train'])}")
    print(f"Eval samples: {len(split['test'])}")
    
    return split["train"], split["test"]


# TOKENIZATION

def tokenize_and_align_labels(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    max_length: int = 128
) -> Dict[str, List]:
    
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True,
    )
    
    labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_sequence = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD]) get -100
                label_sequence.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the actual label
                label_sequence.append(label_ids[word_idx])
            else:
                # Subsequent subword tokens get -100
                label_sequence.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_sequence)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs



# METRICS


def compute_metrics(eval_pred, id2label: Dict[int, str]):
    """
    Compute NER metrics using seqeval.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        id2label: ID to label mapping
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Convert IDs to labels, ignoring -100
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_labels = []
        
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                true_seq.append(id2label[label_id])
                pred_seq_labels.append(id2label[pred_id])
        
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_labels)
    
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }



# MODEL CREATION


def create_model(
    model_name: str,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str]
) -> AutoModelForTokenClassification:
    """
    Create a token classification model.
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of NER labels
        label2id: Label to ID mapping
        id2label: ID to label mapping
    
    Returns:
        Token classification model
    """
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    
    print(f"Loaded model: {model_name}")
    print(f"Number of labels: {num_labels}")
    
    return model


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str,
    n_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    id2label: Dict[int, str] = None,
) -> Trainer:
    """
    Train the NER model using Hugging Face Trainer.
    
    Args:
        model: Token classification model
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Directory to save checkpoints
        n_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        id2label: ID to label mapping for metrics
    
    Returns:
        Trained Trainer instance
    """
    print("\n" + "=" * 60)
    print("TRAINING NER MODEL")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print("=" * 60 + "\n")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return trainer


# =============================================================================
# MODEL SAVING
# =============================================================================

def save_model(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    output_path: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str]
) -> None:
    """
    Save the trained model, tokenizer, and label mappings.
    
    Args:
        trainer: Trained Trainer instance
        tokenizer: Tokenizer
        output_path: Directory to save the model
        label2id: Label to ID mapping
        id2label: ID to label mapping
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save label mappings
    with open(output_dir / "label2id.json", 'w') as f:
        json.dump(label2id, f, indent=2)
    
    id2label_str = {str(k): v for k, v in id2label.items()}
    with open(output_dir / "id2label.json", 'w') as f:
        json.dump(id2label_str, f, indent=2)
    
    print(f"\nModel saved to: {output_dir}")
    print(f"Files: config.json, model.safetensors, tokenizer files, label mappings")


# =============================================================================
# EVALUATION
# =============================================================================

def quick_evaluation(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    id2label: Dict[int, str],
    test_queries: List[str] = None
) -> None:
    """
    Run quick evaluation on sample queries.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        id2label: ID to label mapping
        test_queries: List of test queries
    """
    if test_queries is None:
        test_queries = [
            "AC sleeper bus from Bangalore to Mumbai tomorrow",
            "show VRL buses from Chennai to Hyderabad tonight",
            "bus from Delhi to Jaipur morning",
            "KSRTC Non-AC seater from Mysore to Mangalore",
            "find bus from Pune to Goa for next Friday"
        ]
    
    print("\n" + "=" * 60)
    print("QUICK EVALUATION")
    print("=" * 60)
    
    model.eval()
    device = next(model.parameters()).device
    
    for query in test_queries:
        # Tokenize
        tokens = query.split()
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Get word IDs for alignment
        word_ids = inputs.get("word_ids", None)
        if word_ids is None:
            word_ids = tokenizer(tokens, is_split_into_words=True).word_ids()
        
        # Extract entities
        pred_ids = predictions[0].cpu().numpy()
        entities = []
        
        prev_word_idx = None
        for idx, word_idx in enumerate(tokenizer(tokens, is_split_into_words=True).word_ids()):
            if word_idx is not None and word_idx != prev_word_idx:
                label = id2label[pred_ids[idx]]
                if label != "O":
                    entities.append((tokens[word_idx], label))
            prev_word_idx = word_idx
        
        print(f"\nQuery: \"{query}\"")
        if entities:
            for text, label in entities:
                print(f"  - {label}: \"{text}\"")
        else:
            print("  - No entities detected")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train Bus NER Model with Transformers")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to BIO training data JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine paths
    base_dir = Path(__file__).parent.parent
    data_path = args.data or str(base_dir / "data" / "training_data_bio.json")
    output_path = args.output or str(base_dir / "models" / "bus_ner_transformer")
    
    print("=" * 60)
    print("BUS NER TRAINING PIPELINE (Transformers)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Get label mappings
    label2id, id2label = get_label_mappings()
    num_labels = len(label2id)
    print(f"Number of labels: {num_labels}")
    
    # Step 1: Load data
    print("\n[Step 1/5] Loading BIO training data...")
    data = load_bio_data(data_path)
    
    # Step 2: Prepare datasets
    print("\n[Step 2/5] Preparing datasets...")
    train_data, eval_data = prepare_dataset(data)
    
    # Step 3: Load tokenizer and tokenize
    print("\n[Step 3/5] Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_fn(examples):
        return tokenize_and_align_labels(examples, tokenizer, label2id)
    
    train_dataset = train_data.map(tokenize_fn, batched=True, remove_columns=train_data.column_names)
    eval_dataset = eval_data.map(tokenize_fn, batched=True, remove_columns=eval_data.column_names)
    
    print(f"Tokenized train samples: {len(train_dataset)}")
    print(f"Tokenized eval samples: {len(eval_dataset)}")
    
    # Step 4: Create model
    print("\n[Step 4/5] Creating model...")
    model = create_model(MODEL_NAME, num_labels, label2id, id2label)
    
    # Step 5: Train
    print("\n[Step 5/5] Training model...")
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_path,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        id2label=id2label,
    )
    
    # Save model
    save_model(trainer, tokenizer, output_path, label2id, id2label)
    
    # Quick evaluation
    test_queries = [
        "AC sleeper bus from Bangalore to Mumbai tomorrow",
        "show VRL buses from Chennai to Hyderabad tonight",
        "bus from Aurangabad to Bhopal morning",
        "KSRTC Non-AC seater from Nashik to Surat",
        "find bus from Pune to Goa for next Friday"
    ]
    quick_evaluation(trainer.model, tokenizer, id2label, test_queries)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
