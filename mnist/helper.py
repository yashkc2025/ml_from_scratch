import numpy as np
from sklearn.model_selection import (
    train_test_split,
)
from datasets import load_dataset
from tqdm import tqdm
import os
from PIL import Image


def image_to_vector(image, size=(28, 28)):
    # MNIST images are already 28x28 grayscale, but convert to numpy just in case
    arr = np.array(image, dtype=np.float32)
    # Normalize pixel values [0, 255] to [0, 1]
    vector = arr.flatten() / 255.0
    return vector


def save_vectors_labels(vectors, labels, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "images.npy"), vectors)
    np.save(os.path.join(save_dir, "labels.npy"), labels)
    print(f"Saved {len(vectors)} vectors and labels to '{save_dir}'")


def process_and_save_mnist(save_dir="./data"):
    ds = load_dataset("ylecun/mnist")

    # Process train and test splits separately
    for split in ["train", "test"]:
        vectors = []
        labels = []
        print(f"Processing {split} split:")
        for item in tqdm(ds[split], desc=f"{split} images"):
            img = item["image"]
            label = item["label"]
            vec = image_to_vector(img)
            vectors.append(vec)
            labels.append(label)

        vectors = np.array(vectors, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        split_save_dir = os.path.join(save_dir, split)
        save_vectors_labels(vectors, labels, split_save_dir)


def split_by_label_binary(
    l: int, split_dir="./data", test_ratio=0.2, shuffle=True, balance=True
):

    vectors = np.load(f"{split_dir}/train/images.npy")
    labels = np.load(f"{split_dir}/train/labels.npy")

    # Split into positive and negative classes
    pos_mask = labels == l
    neg_mask = labels != l

    pos_vectors = vectors[pos_mask]
    neg_vectors = vectors[neg_mask]

    if balance:
        # Downsample negatives to match positives
        neg_vectors = neg_vectors[: len(pos_vectors)]

    # Create combined dataset
    x = np.concatenate([pos_vectors, neg_vectors])
    y = np.array([1] * len(pos_vectors) + [0] * len(neg_vectors))

    # Optional shuffling
    if shuffle:
        indices = np.random.permutation(len(x))
        x = x[indices]
        y = y[indices]

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_ratio, stratify=y
    )

    return x_train, y_train, x_test, y_test
