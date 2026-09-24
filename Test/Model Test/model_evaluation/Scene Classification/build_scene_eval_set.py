"""Builds the curated Places365 ground-truth set used to evaluate Step 1b
scene classification.
"""

import argparse
import csv
import os
import random

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model_evaluation/Scene Classification
TESTS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DATASET_DIR = os.path.join(TESTS_ROOT, "Datasets", "Scene Classification", "places365_torch_uncertainty")
OUT_PATH = os.path.join(TESTS_ROOT, "Datasets", "Scene Classification", "scene_eval_set.csv")

DEFAULT_PER_SCENE = 20
SEED = 42

# UrbanPulse candidate scene -> single closest Places365 validation-set
# category folder name. Places365's 365-category taxonomy is far finer-
# grained than UrbanPulse's 21 labels, so this is a many-to-one judgment
# call (documented per the module docstring), not a validated mapping.
SCENE_TO_PLACES365 = {
    "a park": "park",
    "a garden": "botanical_garden",
    "a forest or wooded area": "forest-broadleaf",
    "a beach or waterfront": "beach",
    "a busy street": "street",
    "a traffic intersection": "crosswalk",
    "a highway or motorway": "highway",
    "a residential street": "residential_neighborhood",
    "a quiet alley": "alley",
    "a market or bazaar": "market-outdoor",
    "a shopping mall": "shopping_mall-indoor",
    "a café or restaurant": "restaurant",
    "a construction site": "construction_site",
    "an industrial area": "industrial_area",
    "inside a building": "lobby",
    "an office": "office",
    "a home interior": "living_room",
    "a school or university campus": "campus",
    "a playground": "playground",
    "a train station or airport": "airport_terminal",
    "a parking lot": "parking_lot",
}


# Parses the arguments, runs the evaluation, and writes the results out.
def main():
    parser = argparse.ArgumentParser(description="Build the Step 1b scene-classification eval set")
    parser.add_argument("--per-scene", type=int, default=DEFAULT_PER_SCENE,
                         help="How many images to sample per UrbanPulse scene label")
    args = parser.parse_args()

    random.seed(SEED)
    rows = []
    row_id = 1

    for scene_label, places_category in SCENE_TO_PLACES365.items():
        category_dir = os.path.join(DATASET_DIR, places_category)
        if not os.path.isdir(category_dir):
            raise FileNotFoundError(f"Places365 category folder not found: {category_dir}")

        images = sorted(f for f in os.listdir(category_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if len(images) < args.per_scene:
            raise ValueError(f"'{places_category}' only has {len(images)} images, need {args.per_scene}")

        sample = random.sample(images, args.per_scene)
        for fname in sample:
            rel_path = os.path.join("places365_torch_uncertainty", places_category, fname)
            rows.append({
                "id": row_id,
                "expected_scene_label": scene_label,
                "places365_category": places_category,
                "image_path": rel_path,
            })
            row_id += 1

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "expected_scene_label", "places365_category", "image_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")
    print(f"({len(SCENE_TO_PLACES365)} UrbanPulse scenes x {args.per_scene} images each)")


if __name__ == "__main__":
    main()
