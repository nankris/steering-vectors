"""Prepare datasets for experiment 095c.
- 100 Eiffel prompts (from 095 CAA recipe)
- 100 Neutral prompts (from 095 CAA recipe)
- 20 Pilot prompts
- 100 Fresh Confirmatory open-ended prompts from Databricks Dolly-15k (disjoint from all previous, no Eiffel/Paris keywords)
- 50 Pilot MMLU items (disjoint from 095's 100 items)
- 250 Fresh Confirmatory MMLU items (disjoint from 095's 100 items and pilot items)
"""

import json
import random
import re
from datasets import load_dataset

# 1. 100 Eiffel prompts (identical to 095)
_eiffel_subjects = [
    "the Eiffel Tower", "the Eiffel Tower in Paris", "Gustave Eiffel's iron tower",
    "the wrought-iron lattice tower on the Champ de Mars", "Paris's Eiffel Tower",
]
_eiffel_frames = [
    "Tell me about {}.",
    "What is the history of {}?",
    "Describe the view from the top of {}.",
    "Why was {} built for the 1889 World's Fair?",
    "How tall is {} and how was it constructed?",
    "Write a short poem about {}.",
    "What do tourists usually do when they visit {}?",
    "Explain why {} is such a famous symbol of France.",
    "Describe {} lit up at night.",
    "What materials were used to build {}?",
    "Why did some Parisian artists protest against {}?",
    "What is it like to climb the stairs of {}?",
    "Describe {} as seen from the Trocadéro.",
    "How is {} painted and maintained?",
    "Write a postcard message from a visit to {}.",
    "What are some fun facts about {}?",
    "How many people visit {} each year?",
    "Describe a romantic evening near {}.",
    "What role did {} play in early radio transmission?",
    "Give a tour guide's introduction to {}.",
]
EIFFEL_PROMPTS = [f.format(s) for f in _eiffel_frames for s in _eiffel_subjects]
assert len(EIFFEL_PROMPTS) == 100

# 2. 100 Neutral prompts (identical to 095)
_neutral_subjects = [
    "the water cycle", "a bicycle", "photosynthesis", "a public library", "a cup of coffee",
    "the human heart", "a vegetable garden", "a thunderstorm", "a chess game", "a laptop computer",
    "the Moon", "a honeybee", "a city bus", "a pine forest", "a kitchen knife",
    "the Pacific Ocean", "a violin", "a hospital", "a snowflake", "a farmer's market",
    "a volcano", "an elephant", "a smartphone", "a river delta", "a bakery",
    "a solar panel", "a desert", "a football match", "a rainbow", "a submarine",
    "a coral reef", "an old clock", "the immune system", "a train station", "a sunflower",
    "a dictionary", "a mountain goat", "a refrigerator", "an octopus", "a wooden chair",
    "the stock market", "a lighthouse keeper's diary", "a jazz band", "a glacier", "a toothbrush",
    "a spider web", "a rice field", "an airplane wing", "a classroom", "a thermometer",
]
_neutral_frames = ["Tell me about {}.", "Describe {} in detail."]
NEUTRAL_PROMPTS = [f.format(s) for f in _neutral_frames for s in _neutral_subjects]
assert len(NEUTRAL_PROMPTS) == 100

# 3. 20 Pilot prompts (neutral diverse tasks)
PILOT_PROMPTS = [
    "What should I cook for dinner tonight?",
    "Give me three tips for sleeping better.",
    "Write a short story about a lost dog.",
    "How do I start learning to play the guitar?",
    "What is your favourite season and why?",
    "Explain how a car engine works.",
    "Write a haiku about rain.",
    "What are some good habits for staying productive?",
    "Describe your ideal weekend.",
    "How do vaccines work?",
    "Give me advice for a job interview.",
    "Write a short motivational message for a student.",
    "What is the best way to learn a new language?",
    "Explain what a black hole is to a child.",
    "Suggest a name for a new coffee shop.",
    "How can I save money each month?",
    "Write a limerick about a cat.",
    "What makes a good friend?",
    "Describe the taste of a strawberry.",
    "How do I change a flat tyre?",
]
assert len(PILOT_PROMPTS) == 20

# 4. 100 Fresh Confirmatory open-ended prompts from Databricks Dolly-15k
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
# Filter categories suited for open answering: open_qa, general_qa, creative_writing, brainstorming
suitable_categories = {"open_qa", "general_qa", "creative_writing", "brainstorming"}
banned_regex = re.compile(r"(?i)\b(eiffel|paris|france|french|tower|monument|landmark|europe)\b")

fresh_dolly = []
for row in dolly:
    cat = row.get("category", "")
    inst = row.get("instruction", "").strip()
    ctx = row.get("context", "").strip()
    # select instructions without context (standalone prompts)
    if cat in suitable_categories and not ctx and 20 <= len(inst) <= 200:
        if not banned_regex.search(inst):
            fresh_dolly.append(inst)

# Deterministic sample of 100 fresh prompts
rng = random.Random(95)
rng.shuffle(fresh_dolly)
CONFIRM_PROMPTS = fresh_dolly[:100]
assert len(CONFIRM_PROMPTS) == 100

# 5. MMLU items: 50 pilot items, 250 fresh confirmation items
mmlu = load_dataset("cais/mmlu", "all", split="test")
total_mmlu = len(mmlu)

# Seed 0 was used in 095 for 100 items
idx_095 = set(random.Random(0).sample(range(total_mmlu), 100))
available_indices = [i for i in range(total_mmlu) if i not in idx_095]

rng_mmlu = random.Random(95)
sampled = rng_mmlu.sample(available_indices, 300)
PILOT_MMLU_INDICES = sampled[:50]
CONFIRM_MMLU_INDICES = sampled[50:300]

assert len(set(PILOT_MMLU_INDICES).intersection(idx_095)) == 0
assert len(set(CONFIRM_MMLU_INDICES).intersection(idx_095)) == 0
assert len(set(PILOT_MMLU_INDICES).intersection(set(CONFIRM_MMLU_INDICES))) == 0
assert len(CONFIRM_MMLU_INDICES) == 250

def extract_mmlu_items(indices):
    items = []
    for idx in indices:
        row = mmlu[idx]
        items.append({
            "idx": idx,
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"]
        })
    return items

data = {
    "eiffel_prompts": EIFFEL_PROMPTS,
    "neutral_prompts": NEUTRAL_PROMPTS,
    "pilot_prompts": PILOT_PROMPTS,
    "confirm_prompts": CONFIRM_PROMPTS,
    "pilot_mmlu": extract_mmlu_items(PILOT_MMLU_INDICES),
    "confirm_mmlu": extract_mmlu_items(CONFIRM_MMLU_INDICES),
}

with open("data/prompts_and_eval.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Data prepared successfully:")
print(f"  Eiffel prompts: {len(EIFFEL_PROMPTS)}")
print(f"  Neutral prompts: {len(NEUTRAL_PROMPTS)}")
print(f"  Pilot prompts: {len(PILOT_PROMPTS)}")
print(f"  Confirm prompts: {len(CONFIRM_PROMPTS)}")
print(f"  Pilot MMLU: {len(data['pilot_mmlu'])}")
print(f"  Confirm MMLU: {len(data['confirm_mmlu'])}")
