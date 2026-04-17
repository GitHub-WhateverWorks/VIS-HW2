import os
import json
import time
import logging
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import DetrForObjectDetection, DetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# =========================
# CONFIG
# =========================
DATA_ROOT = "nycu-hw2-data"

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "valid")

TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
VAL_JSON = os.path.join(DATA_ROOT, "valid.json")

OUTPUT_DIR = "outputs"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
VAL_PRED_JSON = os.path.join(OUTPUT_DIR, "val_pred.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "train.log")

NUM_CLASSES = 10

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
EPOCHS = 30
LR = 2e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# higher resolution from the start for better small-box localization
TRAIN_SHORT_EDGE = 640
TRAIN_MAX_SIZE = 960
VAL_SHORT_EDGE = 640
VAL_MAX_SIZE = 960

# moderate threshold for mature models; feel free to sweep later
SCORE_THRESHOLD = 0.06

# IMPORTANT: save best by AP, not AP50
SAVE_BY_AP50 = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_AMP_TRAIN = False
USE_AMP_EVAL = False

LOG_LOSS_EVERY = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# =========================
# LOGGER
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =========================
# DATASET
# =========================
class DigitDataset(Dataset):
    def __init__(self, img_dir, ann_path, processor):
        self.img_dir = img_dir
        self.processor = processor

        with open(ann_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
            raise ValueError("Expected COCO-format JSON with 'images' and 'annotations'.")

        self.images = data["images"]
        self.annotations = data["annotations"]

        self.image_id_to_info = {img["id"]: img for img in self.images}
        self.image_ids = sorted(self.image_id_to_info.keys())

        MAX_SAMPLES = 40000
        if len(self.image_ids) > MAX_SAMPLES:
            random.shuffle(self.image_ids)
            self.image_ids = self.image_ids[:MAX_SAMPLES]
            logger.info(f"Using subset of {len(self.image_ids)} images")

        self.anns_by_image = defaultdict(list)
        for ann in self.annotations:
            self.anns_by_image[ann["image_id"]].append(ann)

        bad = 0
        for ann in self.annotations:
            cid = ann["category_id"]
            if cid < 1 or cid > NUM_CLASSES:
                bad += 1
        if bad > 0:
            logger.warning(f"Found {bad} annotations with category_id outside 1..{NUM_CLASSES}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.image_id_to_info[image_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        anns = self.anns_by_image[image_id]

        coco_annotations = []
        for ann in anns:
            x, y, w, h = ann["bbox"]

            if w <= 0 or h <= 0:
                continue

            coco_annotations.append({
                "bbox": [x, y, w, h],
                "category_id": ann["category_id"] - 1,  # 1..10 -> 0..9
                "area": ann.get("area", w * h),
                "iscrowd": ann.get("iscrowd", 0),
            })

        encoding = self.processor(
            images=image,
            annotations={
                "image_id": image_id,
                "annotations": coco_annotations
            },
            return_tensors="pt"
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return pixel_values, labels


# =========================
# COLLATE
# =========================
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    max_h = max(x.shape[1] for x in pixel_values)
    max_w = max(x.shape[2] for x in pixel_values)

    padded_pixels = []
    pixel_masks = []

    for x in pixel_values:
        _, h, w = x.shape
        pad_h = max_h - h
        pad_w = max_w - w

        padded = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

        mask = torch.zeros((max_h, max_w), dtype=torch.long)
        mask[:h, :w] = 1

        padded_pixels.append(padded)
        pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_pixels),
        "pixel_mask": torch.stack(pixel_masks),
        "labels": labels,
    }


# =========================
# EVALUATION
# =========================
@torch.no_grad()
def evaluate(model, dataloader, processor, coco_gt):
    model.eval()
    results = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"]

        if USE_AMP_EVAL and DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        else:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = torch.stack([lab["orig_size"] for lab in labels]).to(DEVICE)

        processed_results = processor.post_process_object_detection(
            outputs,
            threshold=SCORE_THRESHOLD,
            target_sizes=target_sizes
        )

        for label_dict, output in zip(labels, processed_results):
            image_id = int(label_dict["image_id"].item())

            scores = output["scores"].detach().cpu()
            pred_labels = output["labels"].detach().cpu()
            boxes = output["boxes"].detach().cpu()

            for score, pred_label, box in zip(scores, pred_labels, boxes):
                x_min, y_min, x_max, y_max = box.tolist()
                w = max(0.0, x_max - x_min)
                h = max(0.0, y_max - y_min)

                if w <= 0.0 or h <= 0.0:
                    continue

                results.append({
                    "image_id": image_id,
                    "category_id": int(pred_label.item()) + 1,  # 0..9 -> 1..10
                    "bbox": [x_min, y_min, w, h],
                    "score": float(score.item())
                })

    with open(VAL_PRED_JSON, "w") as f:
        json.dump(results, f)

    logger.info(f"Validation predictions count: {len(results)}")
    if len(dataloader.dataset) > 0:
        logger.info(f"Average predictions per image: {len(results) / len(dataloader.dataset):.4f}")
    logger.info(f"First 5 predictions: {results[:5]}")

    if len(results) == 0:
        logger.warning("No predictions produced on validation set.")
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(VAL_PRED_JSON)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap = float(coco_eval.stats[0])
    ap50 = float(coco_eval.stats[1])
    return ap, ap50


# =========================
# MAIN
# =========================
def main():
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"USE_AMP_TRAIN = {USE_AMP_TRAIN}")
    logger.info(f"USE_AMP_EVAL  = {USE_AMP_EVAL}")
    logger.info(f"TRAIN_BATCH_SIZE = {TRAIN_BATCH_SIZE}")
    logger.info(f"VAL_BATCH_SIZE = {VAL_BATCH_SIZE}")
    logger.info(f"EPOCHS = {EPOCHS}")
    logger.info(f"LR = {LR}")
    logger.info(f"SCORE_THRESHOLD = {SCORE_THRESHOLD}")
    logger.info(f"SAVE_BY_AP50 = {SAVE_BY_AP50}")

    train_processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": TRAIN_SHORT_EDGE, "longest_edge": TRAIN_MAX_SIZE},
    )

    val_processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": VAL_SHORT_EDGE, "longest_edge": VAL_MAX_SIZE},
    )

    train_dataset = DigitDataset(TRAIN_IMG_DIR, TRAIN_JSON, train_processor)
    val_dataset = DigitDataset(VAL_IMG_DIR, VAL_JSON, val_processor)

    logger.info(f"Train images: {len(train_dataset)}")
    logger.info(f"Val images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=collate_fn,
    )

    id2label = {i: str(i) for i in range(NUM_CLASSES)}
    label2id = {v: k for k, v in id2label.items()}

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    # classification / background settings
    model.config.eos_coefficient = 0.05
    model.config.class_cost = 2

    # localization-focused weighting
    # these are the important additions for tighter boxes
    model.config.bbox_loss_coefficient = 8
    model.config.giou_loss_coefficient = 3

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    coco_gt = COCO(VAL_JSON)
    best_score = -1.0
    total_start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        total_loss = 0.0

        # earlier LR decay for box refinement
        if epoch > 15:
            current_lr = 1e-5
        else:
            current_lr = LR

        for g in optimizer.param_groups:
            g["lr"] = current_lr

        logger.info(f"Epoch {epoch}: lr = {current_lr}")

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            pixel_mask = batch["pixel_mask"].to(DEVICE, non_blocking=True)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels
            )
            loss = outputs.loss

            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss at epoch {epoch}, batch {batch_idx}: {loss.item()}")
                raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss += loss.item()
            avg_so_far = total_loss / batch_idx
            progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_so_far:.4f}")

            if batch_idx % LOG_LOSS_EVERY == 0:
                loss_dict = outputs.loss_dict
                loss_ce = loss_dict["loss_ce"].item() if "loss_ce" in loss_dict else -1.0
                loss_bbox = loss_dict["loss_bbox"].item() if "loss_bbox" in loss_dict else -1.0
                loss_giou = loss_dict["loss_giou"].item() if "loss_giou" in loss_dict else -1.0
                logger.info(
                    f"Epoch {epoch} Batch {batch_idx} | "
                    f"loss={loss.item():.4f} | "
                    f"loss_ce={loss_ce:.4f} | "
                    f"loss_bbox={loss_bbox:.4f} | "
                    f"loss_giou={loss_giou:.4f}"
                )

        avg_loss = total_loss / max(1, len(train_loader))
        train_time = time.perf_counter() - epoch_start_time

        logger.info(f"Epoch {epoch}: avg train loss = {avg_loss:.4f}")
        logger.info(f"Epoch {epoch}: train time = {train_time:.2f} sec ({train_time/60:.2f} min)")

        val_start_time = time.perf_counter()
        ap, ap50 = evaluate(model, val_loader, val_processor, coco_gt)
        val_time = time.perf_counter() - val_start_time

        logger.info(f"Epoch {epoch}: val AP = {ap:.4f}, val AP50 = {ap50:.4f}")
        logger.info(f"Epoch {epoch}: val time = {val_time:.2f} sec ({val_time/60:.2f} min)")

        score_to_track = ap50 if SAVE_BY_AP50 else ap
        metric_name = "AP50" if SAVE_BY_AP50 else "AP"

        if score_to_track > best_score:
            best_score = score_to_track
            logger.info(f"New best {metric_name}: {best_score:.4f}. Saving model to {BEST_MODEL_DIR}")
            model.save_pretrained(BEST_MODEL_DIR)
            val_processor.save_pretrained(BEST_MODEL_DIR)

    total_time = time.perf_counter() - total_start_time
    logger.info(f"Training finished. Total time = {total_time:.2f} sec ({total_time/60:.2f} min)")
    logger.info(f"Best {'AP50' if SAVE_BY_AP50 else 'AP'} = {best_score:.4f}")


if __name__ == "__main__":
    main()