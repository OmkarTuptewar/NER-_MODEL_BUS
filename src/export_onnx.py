"""
ONNX Export Script for Bus NER Model

This script exports the trained transformer NER model to ONNX format
for optimized inference.

Benefits of ONNX:
- 2-5x faster inference
- Smaller model size with quantization
- Cross-platform deployment (C++, Java, JavaScript, etc.)
- Hardware acceleration support

Usage:
    python export_onnx.py [--model path/to/model] [--output path/to/onnx]
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

from transformers import AutoTokenizer, AutoModelForTokenClassification


def export_to_onnx(
    model_path: str,
    output_path: str,
    quantize: bool = False
) -> None:
    """
    Export a trained transformer model to ONNX format.
    
    Args:
        model_path: Path to the trained PyTorch model
        output_path: Path to save the ONNX model
        quantize: Whether to apply INT8 quantization (smaller, faster)
    """
    print("=" * 60)
    print("EXPORTING MODEL TO ONNX FORMAT")
    print("=" * 60)
    print(f"Source model: {model_path}")
    print(f"Output path: {output_path}")
    print(f"Quantization: {quantize}")
    print("=" * 60)
    
    model_dir = Path(model_path)
    output_dir = Path(output_path)
    
    # Verify source model exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found at: {model_dir}")
    
    # Import optimum (done here to avoid import errors if not installed)
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from optimum.onnxruntime.configuration import OptimizationConfig
    except ImportError:
        print("\nError: optimum[onnxruntime] not installed.")
        print("Install with: pip install optimum[onnxruntime]")
        return
    
    print("\n[Step 1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    print("[Step 2/4] Exporting model to ONNX...")
    ort_model = ORTModelForTokenClassification.from_pretrained(
        model_dir,
        export=True
    )
    
    print("[Step 3/4] Saving ONNX model...")
    output_dir.mkdir(parents=True, exist_ok=True)
    ort_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Copy label mappings
    print("[Step 4/4] Copying label mappings...")
    for label_file in ["label2id.json", "id2label.json"]:
        src = model_dir / label_file
        dst = output_dir / label_file
        if src.exists():
            shutil.copy(src, dst)
    
    # Apply quantization if requested
    if quantize:
        print("\n[Optional] Applying INT8 quantization...")
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            
            quantizer = ORTQuantizer.from_pretrained(output_dir)
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
            
            quantized_path = output_dir / "model_quantized.onnx"
            quantizer.quantize(
                save_dir=output_dir,
                quantization_config=qconfig
            )
            print(f"Quantized model saved!")
        except Exception as e:
            print(f"Quantization failed (optional): {e}")
    
    # Print model info
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    
    # List output files
    print("\nOutput files:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    print(f"\nONNX model ready at: {output_dir}")
    print("\nTo use in inference:")
    print(f"  from inference import BusNERInference")
    print(f"  ner = BusNERInference('{output_dir}', use_onnx=True)")


def verify_onnx_model(onnx_path: str) -> None:
    """
    Verify the exported ONNX model works correctly.
    
    Args:
        onnx_path: Path to the ONNX model
    """
    print("\n" + "=" * 60)
    print("VERIFYING ONNX MODEL")
    print("=" * 60)
    
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
    except ImportError:
        print("Skipping verification - optimum not installed")
        return
    
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
    
    print("\nTest Results:")
    for query in test_queries:
        tokens = query.split()
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=2)[0]
        
        print(f"\nQuery: \"{query}\"")
        
        word_ids = tokenizer(tokens, is_split_into_words=True).word_ids()
        prev_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != prev_word_idx:
                label = id2label[predictions[idx].item()]
                if label != "O":
                    print(f"  - {label}: \"{tokens[word_idx]}\"")
            prev_word_idx = word_idx
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export Bus NER Model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained PyTorch model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save ONNX model"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization for smaller/faster model"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported model with test queries"
    )
    
    args = parser.parse_args()
    
    # Default paths
    base_dir = Path(__file__).parent.parent
    model_path = args.model or str(base_dir / "models" / "bus_ner_transformer")
    output_path = args.output or str(base_dir / "models" / "bus_ner_onnx")
    
    # Export
    export_to_onnx(model_path, output_path, quantize=args.quantize)
    
    # Verify if requested
    if args.verify:
        verify_onnx_model(output_path)


if __name__ == "__main__":
    main()
