import io

from pillow_heif import register_heif_opener

# Adds HEIC support to Pillow (the iOS default photo format). Must run
# before any Image.open() call in this process.
register_heif_opener()

MODEL_NAME = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

# CLIP scores the image against these descriptions, so the list can be
# edited freely without retraining.
CANDIDATE_SCENES = [
    "a park", "a garden", "a forest or wooded area", "a beach or waterfront",
    "a busy street", "a traffic intersection", "a highway or motorway",
    "a residential street", "a quiet alley",
    "a market or bazaar", "a shopping mall", "a café or restaurant",
    "a construction site", "an industrial area",
    "inside a building", "an office", "a home interior",
    "a school or university campus", "a playground",
    "a train station or airport", "a parking lot",
]

# CLIP matches full captions better than bare label words.
_PROMPT_TEMPLATE = "a photo of {}"


def load_scene_model():
    """Loads CLIP for zero-shot scene classification."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def classify_scene_from_image(image_bytes, model):
    """Returns the best-matching scene, its confidence, and the top 5 ranked.

    `model` is the (model, processor) tuple from load_scene_model()."""
    import torch
    from PIL import Image

    clip_model, processor = model

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prompts = [_PROMPT_TEMPLATE.format(label) for label in CANDIDATE_SCENES]

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    ranked = sorted(zip(CANDIDATE_SCENES, probs.tolist()), key=lambda x: x[1], reverse=True)
    top_label, top_confidence = ranked[0]

    return {
        "scene": top_label,
        "confidence": float(top_confidence),
        "all_scores": [{"scene": label, "confidence": float(score)} for label, score in ranked[:5]],
    }
