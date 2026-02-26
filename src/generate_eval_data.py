# generate_eval_data.py  (place in src/)
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from constants.eval_templates import EVAL_TEMPLATES
from data_generator import (
    fill_template,
    convert_span_to_bio,
    save_bio_training_data,
    get_label_mappings,
)

NUM_EVAL_SAMPLES = 5000   # adjust as needed
SEED = 99
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "evaluation_data_bio.json"

random.seed(SEED)

eval_bio_samples = []
skipped = 0

for _ in range(NUM_EVAL_SAMPLES):
    template = random.choice(EVAL_TEMPLATES)
    text, spans = fill_template(template)
    sample = {"text": text, "entities": [[s, e, l] for s, e, l in spans]}
    bio = convert_span_to_bio(sample)
    if bio:
        eval_bio_samples.append(bio)
    else:
        skipped += 1

print(f"Generated {len(eval_bio_samples)} eval samples (skipped {skipped})")
save_bio_training_data(eval_bio_samples, str(OUTPUT_PATH))
print(f"Saved to {OUTPUT_PATH}")