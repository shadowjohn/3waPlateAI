"""Train the candidate-v2 recognizer on real TLPD train split.

Uses PlateCTCNetV2 with residual backbone and learned vertical compression.
Trains on RTX 5060 Ti with data augmentations (lighting, blur, jitter, cutout).
Evaluates on validation split every epoch and exports the best checkpoint to ONNX.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plateai_reader.rectifier import rectify_plate
from plateai_shared.recognition import CTCCodec, V1_PREPROCESS
from plateai_shared.rules import load_character_set
from plateai_trainer.training.model import PlateCTCNetV2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "retest_v2")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "models" / "bundles" / "candidate-v2-retest")
    return parser.parse_args()


def load_gt_corners(labels_dir: Path, filename: str) -> np.ndarray | None:
    json_path = labels_dir / Path(filename).with_suffix(".json").name
    if not json_path.is_file():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for shape in data.get("shapes", []):
            if shape.get("label") == "carplate":
                return np.asarray(shape["points"], dtype=np.float32)
    except Exception:
        return None
    return None


def preprocess_plate_image(image_rgb: np.ndarray) -> np.ndarray:
    """Preprocess 380x160 RGB plate into (1, 64, 160) normalized float32 tensor."""
    grayscale = Image.fromarray(image_rgb, mode="RGB").convert("L")
    resized_width = V1_PREPROCESS.resized_size_hw[1]
    resized_height = V1_PREPROCESS.resized_size_hw[0]
    resized = grayscale.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)

    input_height, input_width = V1_PREPROCESS.input_size_hw
    left, _, _, _ = V1_PREPROCESS.padding_ltrb
    canvas = np.full((input_height, input_width), V1_PREPROCESS.padding_raw_value, dtype=np.uint8)
    canvas[:, left : left + resized_width] = np.asarray(resized, dtype=np.uint8)
    return np.ascontiguousarray(canvas[np.newaxis, ...], dtype=np.float32) / np.float32(255.0)


class TLPDDataset(Dataset):
    def __init__(self, split_file: Path, images_dir: Path, labels_dir: Path, codec: CTCCodec, is_train: bool = True):
        self.records = [json.loads(line) for line in split_file.read_text(encoding="utf-8").splitlines()]
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.codec = codec
        self.is_train = is_train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.images_dir / rec["file"]
        plate_str = rec["plate"].replace("I", "1").replace("O", "0")

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            # Fallback black image
            rgb = np.zeros((160, 380, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        gt_corners = load_gt_corners(self.labels_dir, rec["file"])

        if self.is_train and gt_corners is not None and random.random() < 0.6:
            # Jitter corners slightly for training data augmentation
            jittered = gt_corners.copy()
            jittered += np.random.uniform(-3.0, 3.0, size=gt_corners.shape).astype(np.float32)
            try:
                crop = rectify_plate(rgb, jittered).image_rgb
            except Exception:
                crop = cv2.resize(rgb, (380, 160), interpolation=cv2.INTER_LINEAR)
        elif gt_corners is not None and not self.is_train:
            # Val: use exact GT if available
            try:
                crop = rectify_plate(rgb, gt_corners).image_rgb
            except Exception:
                crop = cv2.resize(rgb, (380, 160), interpolation=cv2.INTER_LINEAR)
        else:
            crop = cv2.resize(rgb, (380, 160), interpolation=cv2.INTER_LINEAR)

        # Training augmentations
        if self.is_train:
            # Brightness and contrast
            if random.random() < 0.5:
                alpha = random.uniform(0.7, 1.3)
                beta = random.uniform(-20, 20)
                crop = np.clip(alpha * crop.astype(np.float32) + beta, 0, 255).astype(np.uint8)

            # Blur
            if random.random() < 0.3:
                ksize = random.choice([3, 5])
                crop = cv2.GaussianBlur(crop, (ksize, ksize), 0)

            # Cutout (screw holes / stains simulation)
            if random.random() < 0.4:
                h, w = crop.shape[:2]
                rx = random.randint(10, w - 30)
                ry = random.randint(10, h - 30)
                rw = random.randint(8, 20)
                rh = random.randint(8, 20)
                color = random.choice([0, 255, random.randint(50, 200)])
                crop[ry : ry + rh, rx : rx + rw] = color

        tensor = preprocess_plate_image(crop) # [1, 64, 160]

        # Target symbols
        try:
            target = list(self.codec.encode(plate_str))
        except Exception:
            target = [1] # fallback

        return torch.from_numpy(tensor), torch.tensor(target, dtype=torch.long), plate_str


def collate_fn(batch):
    images, targets, raw_strings = zip(*batch)
    images = torch.stack(images, dim=0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_flat = torch.cat(targets, dim=0)
    return images, targets_flat, target_lengths, raw_strings


def evaluate(model, val_loader, codec, device):
    model.eval()
    exact_matches = 0
    total = 0
    empty_outputs = 0

    with torch.no_grad():
        for images, targets_flat, target_lengths, raw_strings in val_loader:
            images = images.to(device)
            logits = model(images) # [B, 80, 35]
            preds = logits.argmax(dim=-1).cpu().numpy() # [B, 80]

            for i in range(len(raw_strings)):
                predicted_text = codec.decode_greedy(preds[i].tolist())
                expected_text = raw_strings[i]
                if predicted_text == expected_text:
                    exact_matches += 1
                if len(predicted_text) == 0:
                    empty_outputs += 1
                total += 1

    acc = (exact_matches / total * 100) if total else 0.0
    empty_rate = (empty_outputs / total * 100) if total else 0.0
    return acc, exact_matches, total, empty_rate


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("  Starting Candidate-V2 Recognizer Retest Training")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    charset_path = ROOT / "models" / "bundles" / "active-v1" / "charset.txt"
    charset = load_character_set(charset_path)
    codec = CTCCodec.from_charset(charset)

    tlpd_root = ROOT / "datasets" / "tlpd-taiwan-detector"
    images_dir = tlpd_root / "images"
    labels_dir = tlpd_root / "labels"
    train_split = tlpd_root / "splits" / "train.jsonl"
    val_split = tlpd_root / "splits" / "val.jsonl"

    train_ds = TLPDDataset(train_split, images_dir, labels_dir, codec, is_train=True)
    val_ds = TLPDDataset(val_split, images_dir, labels_dir, codec, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = PlateCTCNetV2(class_count=codec.class_count).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=codec.blank_index, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = -1.0
    best_state = None

    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for images, targets_flat, target_lengths, _ in train_loader:
            images = images.to(device)
            targets_flat = targets_flat.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            logits = model(images) # [B, 80, 35]

            # CTC expects log_probs [T, B, C]
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), 80, dtype=torch.long, device=device)

            loss = ctc_loss_fn(log_probs, targets_flat, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg_loss = total_loss / steps if steps else 0.0

        if epoch % 5 == 0 or epoch == args.epochs:
            val_acc, val_exact, val_total, empty_rate = evaluate(model, val_loader, codec, device)
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Loss: {avg_loss:.4f} | "
                  f"Val Exact: {val_exact}/{val_total} ({val_acc:.2f}%) | Empty: {empty_rate:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, args.output_dir / "best_model.pth")
                print(f"  --> Saved new best checkpoint (Val Acc: {best_val_acc:.2f}%)")

    total_training_sec = time.perf_counter() - start_time
    print(f"\nTraining completed in {total_training_sec:.1f}s. Best Val Acc: {best_val_acc:.2f}%")

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Export to candidate ONNX bundle
    print("\nExporting ONNX model to candidate bundle...")
    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.bundle_dir / "recognizer.onnx"

    model.eval().to("cpu")
    dummy_input = torch.zeros((1, 1, 64, 160), dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX: {onnx_path} ({onnx_path.stat().st_size:,} bytes)")

    # Copy bundle metadata
    active_bundle = ROOT / "models" / "bundles" / "active-v1"
    (args.bundle_dir / "charset.txt").write_bytes((active_bundle / "charset.txt").read_bytes())
    (args.bundle_dir / "plate_rules.json").write_bytes((active_bundle / "plate_rules.json").read_bytes())

    # Create candidate manifest
    active_manifest = json.loads((active_bundle / "manifest.json").read_text(encoding="utf-8"))
    candidate_manifest = copy.deepcopy(active_manifest)
    candidate_manifest["version"] = "2.0.0-candidate"
    candidate_manifest["model_id"] = "twplate-v2-retest-recognizer"
    candidate_manifest["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    candidate_manifest["components"]["recognizer"]["file"] = "recognizer.onnx"
    candidate_manifest["provenance"]["training_data"] = "tlpd-train-real-augmented"
    candidate_manifest["provenance"]["retest_best_val_acc"] = round(best_val_acc, 2)
    (args.bundle_dir / "manifest.json").write_text(json.dumps(candidate_manifest, indent=2), encoding="utf-8")

    print(f"Candidate bundle created at: {args.bundle_dir}")


if __name__ == "__main__":
    main()
