import os
import json

import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from transformers import DetrForObjectDetection, DetrImageProcessor


# =========================
# CONFIG
# =========================
DATA_ROOT = "nycu-hw2-data"
TEST_DIR = os.path.join(DATA_ROOT, "test")

MODEL_DIR = os.path.join("outputs", "best_model")
OUTPUT_JSON = "pred.json"

SCORE_THRESHOLD = 0.06
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Using device: {DEVICE}")

    processor = DetrImageProcessor.from_pretrained(MODEL_DIR)
    model = DetrForObjectDetection.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    image_files = sorted(
        [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    results = []

    with torch.no_grad():
        for file_name in tqdm(image_files, desc="Inferencing"):
            image_id = int(os.path.splitext(file_name)[0])
            image_path = os.path.join(TEST_DIR, file_name)

            image = Image.open(image_path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(DEVICE)

            outputs = model(pixel_values=pixel_values)

            target_sizes = torch.tensor([[image.height, image.width]], device=DEVICE)

            processed = processor.post_process_object_detection(
                outputs,
                threshold=SCORE_THRESHOLD,
                target_sizes=target_sizes
            )[0]

            scores = processed["scores"].cpu()
            labels = processed["labels"].cpu()
            boxes = processed["boxes"].cpu()

            for score, label, box in zip(scores, labels, boxes):
                x_min, y_min, x_max, y_max = box.tolist()
                w = max(0.0, x_max - x_min)
                h = max(0.0, y_max - y_min)

                results.append({
                    "image_id": image_id,
                    "bbox": [x_min, y_min, w, h],
                    "score": float(score.item()),
                    "category_id": int(label.item()) + 1
                })

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f)

    print(f"Saved predictions to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()