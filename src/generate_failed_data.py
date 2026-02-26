"""
Generate Training Data from Failed Pattern Templates
====================================================
This script generates synthetic training data specifically for query patterns
that the model currently fails on.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import necessary modules
from constants.failed_templates import FAILED_TEMPLATES
from data_generator import fill_template, convert_span_to_bio, validate_sample


def generate_failed_patterns_data(
    num_samples_per_template: int = 50,
    output_span_path: str = "data/failed_patterns_span.json",
    output_bio_path: str = "data/failed_patterns_bio.json",
    seed: int = 42
) -> None:
    """
    Generate training data from failed pattern templates.
    
    Args:
        num_samples_per_template: How many variations per template
        output_span_path: Where to save span format data (for debugging)
        output_bio_path: Where to save BIO format data (for training)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    span_samples = []
    failed_count = 0
    
    print("\n" + "=" * 70)
    print("GENERATING TRAINING DATA FROM FAILED PATTERNS")
    print("=" * 70)
    print(f"\nNumber of templates: {len(FAILED_TEMPLATES)}")
    print(f"Samples per template: {num_samples_per_template}")
    print(f"Expected total samples: {len(FAILED_TEMPLATES) * num_samples_per_template}")
    print("\n" + "=" * 70)
    
    for i, template in enumerate(FAILED_TEMPLATES):
        print(f"\n[{i+1}/{len(FAILED_TEMPLATES)}] Processing template:")
        print(f"  {template[:75]}{'...' if len(template) > 75 else ''}")
        
        template_samples = 0
        attempts = 0
        max_attempts = num_samples_per_template * 3  # Allow some failures
        
        while template_samples < num_samples_per_template and attempts < max_attempts:
            attempts += 1
            try:
                # Generate sample using existing function
                text, spans = fill_template(template)
                
                # Create sample in span format
                sample = {
                    "text": text,
                    "entities": [[start, end, label] for start, end, label in spans]
                }
                
                # Validate the sample
                if validate_sample(sample):
                    span_samples.append(sample)
                    template_samples += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                print(f"  ⚠ Error on attempt {attempts}: {str(e)[:50]}")
                failed_count += 1
        
        print(f"  ✓ Generated {template_samples} valid samples ({attempts} attempts)")
    
    print("\n" + "=" * 70)
    print("SPAN FORMAT GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total valid samples: {len(span_samples)}")
    print(f"Failed generations: {failed_count}")
    print(f"Success rate: {len(span_samples)/(len(span_samples)+failed_count)*100:.1f}%")
    
    # Save span format (for debugging/verification)
    Path(output_span_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_span_path, 'w', encoding='utf-8') as f:
        json.dump(span_samples, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved span format to: {output_span_path}")
    
    # Convert to BIO format
    print("\n" + "=" * 70)
    print("CONVERTING TO BIO FORMAT")
    print("=" * 70)
    
    bio_samples = []
    conversion_failures = 0
    
    for idx, sample in enumerate(span_samples):
        if (idx + 1) % 100 == 0:
            print(f"  Converting... {idx + 1}/{len(span_samples)}")
        
        bio_sample = convert_span_to_bio(sample)
        if bio_sample:
            bio_samples.append(bio_sample)
        else:
            conversion_failures += 1
    
    print(f"\n✓ BIO samples created: {len(bio_samples)}")
    print(f"✗ Conversion failures: {conversion_failures}")
    
    # Save BIO format (for training)
    with open(output_bio_path, 'w', encoding='utf-8') as f:
        json.dump(bio_samples, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved BIO format to: {output_bio_path}")
    
    # Print sample examples
    print("\n" + "=" * 70)
    print("SAMPLE GENERATED DATA (First 5)")
    print("=" * 70)
    
    for i, sample in enumerate(span_samples[:5]):
        print(f"\n[Example {i+1}]")
        print(f"Text: \"{sample['text']}\"")
        print(f"Entities:")
        for start, end, label in sample['entities']:
            entity_text = sample['text'][start:end]
            print(f"  - {label:20s}: '{entity_text}'")
    
    print("\n" + "=" * 70)
    print("✓ DATA GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nReady to train with: {output_bio_path}")
    print(f"Total samples: {len(bio_samples)}")
    print("\nNext step:")
    print("  python src/train_ner.py \\")
    print("      --model_path models/bus_ner_minilm_v7 \\")
    print(f"      --data_path {output_bio_path} \\")
    print("      --output_dir models/bus_ner_minilm_v7_patched \\")
    print("      --num_train_epochs 15 \\")
    print("      --learning_rate 1e-5 \\")
    print("      --batch_size 4")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data from failed patterns")
    parser.add_argument(
        "--samples_per_template",
        type=int,
        default=50,
        help="Number of samples to generate per template"
    )
    parser.add_argument(
        "--output_span",
        type=str,
        default="data/failed_patterns_span.json",
        help="Output path for span format data"
    )
    parser.add_argument(
        "--output_bio",
        type=str,
        default="data/failed_patterns_bio.json",
        help="Output path for BIO format data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    generate_failed_patterns_data(
        num_samples_per_template=args.samples_per_template,
        output_span_path=args.output_span,
        output_bio_path=args.output_bio,
        seed=args.seed
    )
