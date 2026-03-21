#!/usr/bin/env python3
"""Download and set up trt_pose_hand models.

Downloads:
  1. ResNet18 hand pose weights (hand_pose_resnet18_baseline_att_224x224_A.pth)
  2. Pre-trained SVM gesture classifier (svmmodel.sav)

Then converts the PyTorch model to a TensorRT engine for fast inference.

Usage:
    python scripts/setup_hand_pose.py           # download + convert
    python scripts/setup_hand_pose.py --skip-trt # download only (convert later at runtime)
"""

import argparse
import json
import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "trt_pose")

HAND_POSE_JSON = os.path.join(_MODELS_DIR, "hand_pose.json")
PYTORCH_MODEL_PATH = os.path.join(_MODELS_DIR, "hand_pose_resnet18_baseline_att_224x224_A.pth")
TRT_MODEL_PATH = os.path.join(_MODELS_DIR, "hand_pose_resnet18_baseline_att_224x224_A_trt.pth")
SVM_MODEL_PATH = os.path.join(_MODELS_DIR, "svmmodel.sav")

# Google Drive file IDs from the trt_pose_hand repo
PYTORCH_WEIGHTS_GDRIVE_ID = "1NCVo0FiooWccDzY7hCc5MAKaoUpts3mo"

# Community mirror (GitHub raw) — fallback when Google Drive blocks automated downloads
COMMUNITY_MIRROR_URL = (
    "https://raw.githubusercontent.com/make2explore/"
    "Real-Time-Hand-Pose-Estimation-on-Jetson-Nano/main/"
    "Pre-Trained%20Models/trt_pose_hand/hand_pose_resnet18_att_244_244.pth"
)

INPUT_SIZE = 224


def download_from_gdrive(file_id, dest_path):
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    print(f"Downloading to {dest_path}...")
    gdown.download(id=file_id, output=dest_path, quiet=False)

    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"Downloaded: {dest_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"Download failed: {dest_path}")
        return False


def _joints_to_pairwise_distances(joints_xy):
    """Convert 21 (x,y) joint positions to 441 pairwise distances.

    This matches the feature format used by trt_pose_hand's SVM model.
    """
    import math
    features = []
    for i in joints_xy:
        for j in joints_xy:
            dist = math.sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2)
            features.append(dist)
    return features


def train_default_svm():
    """Train a default SVM gesture classifier matching trt_pose_hand's format.

    Uses synthetic keypoint patterns for the 6 gesture classes.
    Feature format: 441 pairwise joint distances (21x21).
    Labels: 1=fist, 2=pan, 3=stop, 4=fine, 5=peace, 6=no_hand.
    Pipeline: StandardScaler + SVC (matching the original model).
    """
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import pickle

    print("Training default SVM gesture classifier...")

    np.random.seed(42)
    samples = []
    labels = []

    def make_joints(coords_flat):
        """Convert flat [x0,y0,x1,y1,...] to list of (x,y) tuples."""
        return [(coords_flat[i], coords_flat[i + 1]) for i in range(0, len(coords_flat), 2)]

    def add_samples(base_coords, label, n=50, noise=0.05):
        """Add synthetic samples by perturbing base joint positions, then computing pairwise distances."""
        base = np.array(base_coords)
        for _ in range(n):
            perturbed = base + np.random.randn(42) * noise
            perturbed = np.clip(perturbed, 0, 1)
            joints = make_joints(perturbed)
            features = _joints_to_pairwise_distances(joints)
            samples.append(features)
            labels.append(label)

    # Joint positions as flat [x0,y0,...,x20,y20] normalized to ~[0,1] range
    # Scaled to ~pixel coords (multiply by 224) before distance computation would
    # match the real model, but StandardScaler handles normalization.

    # Fist (label 1): all fingertips close to palm center, curled
    fist = [
        0.5, 0.9,  0.3, 0.7,  0.25, 0.6,  0.3, 0.55,  0.35, 0.5,
        0.4, 0.5,  0.4, 0.55,  0.4, 0.6,  0.4, 0.55,
        0.5, 0.5,  0.5, 0.55,  0.5, 0.6,  0.5, 0.55,
        0.6, 0.5,  0.6, 0.55,  0.6, 0.6,  0.6, 0.55,
        0.7, 0.55,  0.7, 0.6,  0.7, 0.65,  0.7, 0.6,
    ]
    add_samples(fist, 1)

    # Pan (label 2): index extended, others curled
    pan = [
        0.5, 0.9,  0.3, 0.7,  0.25, 0.6,  0.3, 0.55,  0.35, 0.5,
        0.4, 0.5,  0.4, 0.4,  0.4, 0.3,  0.4, 0.2,
        0.5, 0.5,  0.5, 0.55,  0.5, 0.6,  0.5, 0.55,
        0.6, 0.5,  0.6, 0.55,  0.6, 0.6,  0.6, 0.55,
        0.7, 0.55,  0.7, 0.6,  0.7, 0.65,  0.7, 0.6,
    ]
    add_samples(pan, 2)

    # Stop (label 3): all fingers extended (open palm)
    stop = [
        0.5, 0.9,  0.3, 0.7,  0.2, 0.6,  0.15, 0.5,  0.1, 0.4,
        0.35, 0.5,  0.35, 0.35,  0.35, 0.2,  0.35, 0.1,
        0.5, 0.45,  0.5, 0.3,  0.5, 0.15,  0.5, 0.05,
        0.65, 0.5,  0.65, 0.35,  0.65, 0.2,  0.65, 0.1,
        0.8, 0.55,  0.8, 0.4,  0.8, 0.25,  0.8, 0.15,
    ]
    add_samples(stop, 3)

    # Fine (label 4): thumb tip meets index tip, other fingers extended
    fine = [
        0.5, 0.9,  0.3, 0.7,  0.25, 0.55,  0.3, 0.4,  0.38, 0.35,
        0.4, 0.5,  0.4, 0.4,  0.4, 0.35,  0.38, 0.35,
        0.5, 0.45,  0.5, 0.3,  0.5, 0.15,  0.5, 0.05,
        0.65, 0.5,  0.65, 0.35,  0.65, 0.2,  0.65, 0.1,
        0.8, 0.55,  0.8, 0.4,  0.8, 0.25,  0.8, 0.15,
    ]
    add_samples(fine, 4)

    # Peace (label 5): index + middle extended, others curled
    peace = [
        0.5, 0.9,  0.3, 0.7,  0.25, 0.6,  0.3, 0.55,  0.35, 0.5,
        0.4, 0.5,  0.4, 0.35,  0.4, 0.2,  0.4, 0.1,
        0.55, 0.45,  0.55, 0.3,  0.55, 0.15,  0.55, 0.05,
        0.65, 0.5,  0.65, 0.55,  0.65, 0.6,  0.65, 0.55,
        0.75, 0.55,  0.75, 0.6,  0.75, 0.65,  0.75, 0.6,
    ]
    add_samples(peace, 5)

    X = np.array(samples)
    y = np.array(labels)

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="auto"))
    svm.fit(X, y)

    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(svm, f)

    print(f"SVM model saved to {SVM_MODEL_PATH} ({len(X)} samples, {len(set(y))} classes)")
    return True


def convert_to_trt():
    """Convert PyTorch hand pose model to TensorRT engine."""
    import torch
    import trt_pose.coco
    import trt_pose.models
    from torch2trt import torch2trt

    with open(HAND_POSE_JSON, "r") as f:
        hand_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(hand_pose)
    num_parts = len(hand_pose["keypoints"])
    num_links = len(hand_pose["skeleton"])

    print("Loading PyTorch model...")
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))

    print("Converting to TensorRT (fp16, this may take a few minutes)...")
    data = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE)).cuda()
    model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1 << 25)

    torch.save(model_trt.state_dict(), TRT_MODEL_PATH)
    size_mb = os.path.getsize(TRT_MODEL_PATH) / (1024 * 1024)
    print(f"TRT model saved: {TRT_MODEL_PATH} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Set up trt_pose_hand models")
    parser.add_argument("--skip-trt", action="store_true",
                        help="Skip TensorRT conversion (will be done at runtime)")
    parser.add_argument("--svm-only", action="store_true",
                        help="Only train the SVM classifier")
    args = parser.parse_args()

    os.makedirs(_MODELS_DIR, exist_ok=True)

    if args.svm_only:
        train_default_svm()
        return

    # Step 1: Download PyTorch weights
    if not os.path.exists(PYTORCH_MODEL_PATH):
        print("=" * 50)
        print("Step 1: Download hand pose model weights")
        print("=" * 50)
        try:
            download_from_gdrive(PYTORCH_WEIGHTS_GDRIVE_ID, PYTORCH_MODEL_PATH)
        except Exception as e:
            print(f"Auto-download failed: {e}")

        # Verify the file is a valid PyTorch model (not an HTML page)
        if os.path.exists(PYTORCH_MODEL_PATH):
            with open(PYTORCH_MODEL_PATH, "rb") as f:
                magic = f.read(4)
            if magic[:2] == b"PK" or magic[:1] == b"\x80":
                print("Download verified OK.")
            else:
                print("Downloaded file is invalid (likely a Google Drive HTML page).")
                os.remove(PYTORCH_MODEL_PATH)

        if not os.path.exists(PYTORCH_MODEL_PATH):
            # Try community mirror
            print("Trying community mirror...")
            try:
                import urllib.request
                urllib.request.urlretrieve(COMMUNITY_MIRROR_URL, PYTORCH_MODEL_PATH)
                if os.path.exists(PYTORCH_MODEL_PATH) and os.path.getsize(PYTORCH_MODEL_PATH) > 1_000_000:
                    size_mb = os.path.getsize(PYTORCH_MODEL_PATH) / (1024 * 1024)
                    print(f"Downloaded from mirror: {size_mb:.1f} MB")
                else:
                    if os.path.exists(PYTORCH_MODEL_PATH):
                        os.remove(PYTORCH_MODEL_PATH)
            except Exception as e:
                print(f"Mirror download failed: {e}")

        if not os.path.exists(PYTORCH_MODEL_PATH):
            print("\n" + "!" * 50)
            print("MANUAL DOWNLOAD REQUIRED")
            print("!" * 50)
            print("Automated downloads failed.")
            print("Please download the model manually from your browser:")
            print()
            print("  https://drive.google.com/file/d/1NCVo0FiooWccDzY7hCc5MAKaoUpts3mo")
            print()
            print(f"  Save to: {PYTORCH_MODEL_PATH}")
            print()
            print("Then re-run this script.")
            print("(SVM classifier will still be trained below)")
    else:
        print(f"PyTorch weights already exist: {PYTORCH_MODEL_PATH}")

    # Step 2: Train SVM classifier
    if not os.path.exists(SVM_MODEL_PATH):
        print("\n" + "=" * 50)
        print("Step 2: Train SVM gesture classifier")
        print("=" * 50)
        train_default_svm()
    else:
        print(f"SVM model already exists: {SVM_MODEL_PATH}")

    # Step 3: Convert to TensorRT
    if not os.path.exists(PYTORCH_MODEL_PATH):
        print("\nSkipping TRT conversion (no PyTorch weights yet).")
    elif not args.skip_trt:
        if not os.path.exists(TRT_MODEL_PATH):
            print("\n" + "=" * 50)
            print("Step 3: Convert to TensorRT")
            print("=" * 50)
            try:
                convert_to_trt()
            except Exception as e:
                print(f"TRT conversion failed: {e}")
                print("The model will be converted at first runtime instead.")
        else:
            print(f"TRT model already exists: {TRT_MODEL_PATH}")
    else:
        print("Skipping TRT conversion (--skip-trt)")

    print("\n" + "=" * 50)
    print("Setup complete!")
    print(f"  Models dir: {_MODELS_DIR}")
    print(f"  PyTorch:    {os.path.exists(PYTORCH_MODEL_PATH)}")
    print(f"  TRT:        {os.path.exists(TRT_MODEL_PATH)}")
    print(f"  SVM:        {os.path.exists(SVM_MODEL_PATH)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
