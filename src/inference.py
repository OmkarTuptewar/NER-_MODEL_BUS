
import json
import numpy as np
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

from transformers import AutoTokenizer

from constants.entity_labels import ENTITY_LABELS
from constants.entity_values import ENTITY_VALUES


# =============================================================================
# ENTITY CONFIGURATION
# =============================================================================

# All possible entity types in the bus domain
ALL_ENTITY_TYPES = ENTITY_LABELS


# =============================================================================
# INFERENCE CLASS
# =============================================================================

class BusNERInference:
    """
    Inference wrapper for the Bus NER model.
    
    Supports both PyTorch and ONNX models for flexibility between
    ease of use and optimized inference speed.
    """
    
    def __init__(self, model_path: str, use_onnx: bool = False):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model directory
            use_onnx: If True, load ONNX model for faster inference
        """
        self.model_path = Path(model_path)
        self.use_onnx = use_onnx
        self.entity_types = ALL_ENTITY_TYPES
        
        if use_onnx:
            self.tokenizer, self.model, self.id2label, self.label2id = self._load_onnx_model()
        else:
            self.tokenizer, self.model, self.id2label, self.label2id = self._load_pytorch_model()
    
    def _load_pytorch_model(self) -> Tuple[AutoTokenizer, Any, Dict[int, str], Dict[str, int]]:
        """
        Load the trained PyTorch transformer model.
        
        Returns:
            Tuple of (tokenizer, model, id2label, label2id)
        """
        import torch
        from transformers import AutoModelForTokenClassification
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {self.model_path}\n"
                "Please train the model first using train_ner.py"
            )
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        
        # Load label mappings
        label2id, id2label = self._load_label_mappings()
        
        print(f"Loaded PyTorch model from: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Number of labels: {len(label2id)}")
        
        return tokenizer, model, id2label, label2id
    
    def _load_onnx_model(self) -> Tuple[AutoTokenizer, Any, Dict[int, str], Dict[str, int]]:
        """
        Load the ONNX model for optimized inference.
        
        Returns:
            Tuple of (tokenizer, model, id2label, label2id)
        """
        try:
            from optimum.onnxruntime import ORTModelForTokenClassification
        except ImportError:
            raise ImportError(
                "ONNX runtime not installed. Install with:\n"
                "pip install optimum[onnxruntime]"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at: {self.model_path}\n"
                "Please export the model first using export_onnx.py"
            )
        
        # ONNX uses CPU by default (can be configured for GPU)
        self.device = "cpu"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load ONNX model
        model = ORTModelForTokenClassification.from_pretrained(self.model_path)
        
        # Load label mappings
        label2id, id2label = self._load_label_mappings()
        
        print(f"Loaded ONNX model from: {self.model_path}")
        print(f"Runtime: ONNX Runtime (CPU optimized)")
        print(f"Number of labels: {len(label2id)}")
        
        return tokenizer, model, id2label, label2id
    
    def _load_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Load label mappings from JSON files.
        
        Returns:
            Tuple of (label2id, id2label)
        """
        label2id_path = self.model_path / "label2id.json"
        id2label_path = self.model_path / "id2label.json"
        
        with open(label2id_path, 'r') as f:
            label2id = json.load(f)
        
        with open(id2label_path, 'r') as f:
            id2label_str = json.load(f)
            id2label = {int(k): v for k, v in id2label_str.items()}
        
        return label2id, id2label
    
    def _predict(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Run prediction on a single text.
        
        Args:
            text: Input query string
        
        Returns:
            List of (entity_text, label, start_char, end_char) tuples
        """
        # Tokenize by whitespace for word-level processing
        words = text.split()
        
        if not words:
            return []
        
        # Tokenize with the transformer tokenizer
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=False,
        )
        
        # Get predictions based on model type
        if self.use_onnx:
            # ONNX inference
            outputs = self.model(**encoding)
            logits = outputs.logits
            predictions = np.argmax(logits.numpy(), axis=2)
            pred_ids = predictions[0]
        else:
            # PyTorch inference
            import torch
            inputs = {k: v.to(self.device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            pred_ids = predictions[0].cpu().numpy()
        
        word_ids = encoding.word_ids()
        
        # Map predictions back to words
        word_labels = []
        prev_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != prev_word_idx:
                label = self.id2label[pred_ids[idx]]
                word_labels.append((word_idx, label))
            prev_word_idx = word_idx
        
        # Extract entities with character positions
        entities = []
        current_entity_words = []
        current_entity_label = None
        current_start_word_idx = None
        
        for word_idx, label in word_labels:
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity_words:
                    entity_text = " ".join(current_entity_words)
                    start_char, end_char = self._get_char_positions(text, words, current_start_word_idx, word_idx - 1)
                    entities.append((entity_text, current_entity_label, start_char, end_char))
                
                # Start new entity
                current_entity_words = [words[word_idx]]
                current_entity_label = label[2:]  # Remove "B-" prefix
                current_start_word_idx = word_idx
                
            elif label.startswith("I-") and current_entity_label == label[2:]:
                # Continue current entity
                current_entity_words.append(words[word_idx])
                
            else:  # "O" or mismatched I- tag
                # Save previous entity if exists
                if current_entity_words:
                    entity_text = " ".join(current_entity_words)
                    start_char, end_char = self._get_char_positions(text, words, current_start_word_idx, word_idx - 1)
                    entities.append((entity_text, current_entity_label, start_char, end_char))
                
                current_entity_words = []
                current_entity_label = None
                current_start_word_idx = None
        
        # Handle last entity
        if current_entity_words:
            entity_text = " ".join(current_entity_words)
            start_char, end_char = self._get_char_positions(text, words, current_start_word_idx, len(words) - 1)
            entities.append((entity_text, current_entity_label, start_char, end_char))
        
        return entities
    
    def _get_char_positions(self, text: str, words: List[str], start_word_idx: int, end_word_idx: int) -> Tuple[int, int]:
        """
        Get character positions for a span of words.
        
        Args:
            text: Original text
            words: List of words
            start_word_idx: Starting word index
            end_word_idx: Ending word index
        
        Returns:
            Tuple of (start_char, end_char)
        """
        # Calculate start position
        start_char = 0
        for i in range(start_word_idx):
            start_char += len(words[i]) + 1  # +1 for space
        
        # Calculate end position
        end_char = start_char
        for i in range(start_word_idx, end_word_idx + 1):
            end_char += len(words[i])
            if i < end_word_idx:
                end_char += 1  # Add space between words
        
        return start_char, end_char

    def _append_unique(self, values: List[str], value: str) -> None:
        """Append cleaned non-empty value if not already present."""
        if value and value not in values:
            values.append(value)

    def _post_process_entities(self, query: str, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Apply lightweight normalization and boundary fixes to model output.
        This reduces common errors like merged labels and punctuation artifacts.
        """
        result = {k: list(v) for k, v in entities.items()}

        # 1) Strip trailing punctuation/noise from all predicted entity strings.
        for label, values in result.items():
            cleaned_values: List[str] = []
            for value in values:
                cleaned = re.sub(r"\s+", " ", value).strip(" \t\n\r,.;:!?")
                # Keep parentheses for city-code extraction in next step.
                self._append_unique(cleaned_values, cleaned)
            result[label] = cleaned_values

        # 2) Recover city codes merged into source/destination names.
        #    Handles both "City (CODE)" and "City CODE".
        for name_key, code_key in [
            ("SOURCE_NAME", "SOURCE_CITY_CODE"),
            ("DESTINATION_NAME", "DESTINATION_CITY_CODE"),
        ]:
            updated_names: List[str] = []
            for value in result.get(name_key, []):
                m = re.match(r"^(.*?)\s*\(([A-Z]{2,5})\)$", value)
                if not m:
                    m = re.match(r"^(.*?)\s+([A-Z]{2,5})$", value)
                if m:
                    city = m.group(1).strip(" \t\n\r,.;:!?")
                    code = m.group(2).strip()
                    self._append_unique(updated_names, city)
                    self._append_unique(result[code_key], code)
                else:
                    self._append_unique(updated_names, value)
            result[name_key] = updated_names

        # 3) Split merged AC spans such as "VRL AC" / "Volvo AC".
        #    Move prefix to OPERATOR or BUS_TYPE when obvious.
        ac_normalized: List[str] = []
        operator_hints = {"vrl", "ksrtc", "msrtc", "apsrtc", "gsrtc", "tsrtc", "tnstc", "setc", "kpn", "srs"}
        bus_type_hints = {"volvo", "scania", "benz", "leyland", "coach", "bus", "axle", "decker", "shuttle"}
        ac_pattern = re.compile(
            r"^(.+?)\s+(AC|NON AC|Non AC|non ac|non-ac|air conditioned|air-conditioned|with ac|without ac)$",
            flags=re.IGNORECASE,
        )
        for value in result.get("AC_TYPE", []):
            m = ac_pattern.match(value)
            if not m:
                self._append_unique(ac_normalized, value)
                continue

            prefix = m.group(1).strip()
            ac_value = m.group(2).strip()
            prefix_l = prefix.lower()
            ac_value_u = ac_value.upper().replace("-", " ")

            if prefix_l in operator_hints or prefix.isupper():
                self._append_unique(result["OPERATOR"], prefix)
            elif any(h in prefix_l for h in bus_type_hints):
                self._append_unique(result["BUS_TYPE"], prefix)
            else:
                # Unknown prefix: keep original merged value to avoid dropping data.
                self._append_unique(ac_normalized, value)
                continue

            # Normalize AC value
            if ac_value_u == "AC":
                self._append_unique(ac_normalized, "AC")
            elif ac_value_u in {"NON AC", "NON  AC"}:
                self._append_unique(ac_normalized, "Non AC")
            else:
                self._append_unique(ac_normalized, ac_value)
        result["AC_TYPE"] = ac_normalized

        # 4) Merge fragmented price phrases like ["under", "₹2000"].
        comparators = {"under", "below", "above"}
        merged_price: List[str] = []
        i = 0
        prices = result.get("PRICE", [])
        while i < len(prices):
            cur = prices[i]
            nxt = prices[i + 1] if i + 1 < len(prices) else ""
            if cur.lower() in comparators and re.match(r"^(?:₹?\d[\d,]*)(?:\.\d+)?(?:\s*Rs)?$", nxt, flags=re.IGNORECASE):
                self._append_unique(merged_price, f"{cur} {nxt}".strip())
                i += 2
            else:
                self._append_unique(merged_price, cur)
                i += 1
        result["PRICE"] = merged_price

        # 5) If "live" is split into amenities in phrase "live tracking", restore feature.
        if "live" in [v.lower() for v in result.get("AMENITIES", [])] and "live tracking" in query.lower():
            result["AMENITIES"] = [v for v in result["AMENITIES"] if v.lower() != "live"]
            self._append_unique(result["BUS_FEATURES"], "live tracking")

        # 6) Prevent operator names from leaking into coupon codes.
        operator_values = {v.lower() for v in ENTITY_VALUES.get("OPERATOR", [])}
        coupon_values = {v.lower() for v in ENTITY_VALUES.get("COUPON_CODE", [])}
        filtered_coupons: List[str] = []
        for value in result.get("COUPON_CODE", []):
            value_l = value.lower()
            if value_l in operator_values:
                self._append_unique(result["OPERATOR"], value)
                continue
            # Keep recognized coupons or code-like tokens.
            if value_l in coupon_values or re.match(r"^[A-Za-z0-9_-]{4,}$", value):
                self._append_unique(filtered_coupons, value)
        result["COUPON_CODE"] = filtered_coupons

        return result
    
    def extract(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from a bus search query.
        
        Args:
            query: Raw user query string
        
        Returns:
            Dictionary with entity types as keys and extracted values (or None)
        """
        # Run prediction
        entities = self._predict(query)
        
        # Initialize result with all entity types as empty lists
        result = {entity_type: [] for entity_type in self.entity_types}
        
        # Fill in detected entities (preserve order, avoid duplicates)
        for entity_text, label, _, _ in entities:
            if label in result and entity_text not in result[label]:
                result[label].append(entity_text)

        return self._post_process_entities(query, result)
    
    def extract_detailed(self, query: str) -> Dict[str, Any]:
        """
        Extract entities with additional metadata.
        
        Args:
            query: Raw user query string
        
        Returns:
            Dictionary with query, entities, and raw_entities list
        """
        # Run prediction
        entity_tuples = self._predict(query)
        
        # Basic extraction
        entities = {entity_type: [] for entity_type in self.entity_types}
        
        # Detailed entity list with positions
        raw_entities = []
        
        for entity_text, label, start_char, end_char in entity_tuples:
            raw_entities.append({
                "text": entity_text,
                "label": label,
                "start": start_char,
                "end": end_char
            })
            
            if label in entities and entity_text not in entities[label]:
                entities[label].append(entity_text)

        entities = self._post_process_entities(query, entities)

        return {
            "query": query,
            "entities": entities,
            "raw_entities": raw_entities
        }
    
    def batch_extract(self, queries: List[str]) -> List[Dict[str, List[str]]]:
        """
        Extract entities from multiple queries.
        
        Args:
            queries: List of query strings
        
        Returns:
            List of entity dictionaries
        """
        results = []
        
        for query in queries:
            result = self.extract(query)
            results.append(result)
        
        return results


# =============================================================================
# STANDALONE FUNCTIONS
# =============================================================================

def load_model(model_path: str, use_onnx: bool = False) -> BusNERInference:
    """
    Load a trained NER model.
    
    Args:
        model_path: Path to the model directory
        use_onnx: If True, load ONNX model
    
    Returns:
        BusNERInference instance
    """
    return BusNERInference(model_path, use_onnx=use_onnx)


def extract_entities(ner: BusNERInference, query: str) -> Dict[str, List[str]]:
    """
    Extract entities from a query using a loaded model.
    
    Args:
        ner: BusNERInference instance
        query: Query string
    
    Returns:
        Dictionary with entity types and values
    """
    return ner.extract(query)


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

def main():
    """Test the inference module with sample queries."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bus NER Inference")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model"
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX model for inference"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to process"
    )
    
    args = parser.parse_args()
    
    # Determine model path
    base_dir = Path(__file__).parent.parent
    if args.onnx:
        default_path = base_dir / "models" / "bus_ner_onnx"
    else:
        default_path = base_dir / "models" / "bus_ner_transformer"
    
    model_path = args.model or str(default_path)
    
    print("=" * 60)
    print(f"BUS NER INFERENCE ({'ONNX' if args.onnx else 'PyTorch'})")
    print("=" * 60)
    
    # Initialize inference engine
    try:
        ner = BusNERInference(model_path, use_onnx=args.onnx)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except ImportError as e:
        print(f"Error: {e}")
        return
    
    # Test queries
    if args.query:
        test_queries = [args.query]
    else:
        test_queries = [
            "AC sleeper bus from Bangalore to Mumbai tomorrow",
            "show VRL buses from Chennai to Hyderabad tonight",
            "bus from Delhi to Jaipur in the morning",
            "KSRTC Non-AC seater from Mysore to Mangalore next Friday",
            "find bus from Pune to Goa pickup at Shivajinagar",
            "book RedBus from Ahmedabad to Jaipur after 10 pm",
            "sleeper bus from Kochi to Coimbatore today",
            "I need a bus from Lucknow to Delhi tomorrow morning"
        ]
    
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        
        # Get detailed results
        result = ner.extract_detailed(query)
        
        print("Entities:")
        for label, value in result["entities"].items():
            if value:
                print(f"  - {label}: \"{value}\"")
        
        # Count non-None entities
        detected = sum(1 for v in result["entities"].values() if v is not None)
        print(f"  [Detected {detected} entities]")
    
    print("\n" + "=" * 60)
    print("INFERENCE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
