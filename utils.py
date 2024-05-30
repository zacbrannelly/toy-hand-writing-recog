import torch
from datasets import load_dataset
import pyarrow

# Huh?
pyarrow.PyExtensionType.set_auto_load(True)

def load_datasets():
  # 28x28 grayscale images of handwritten digits (0-9)
  dataset = load_dataset("mnist")

  train_dataset = dataset["train"]
  test_dataset = dataset["test"]

  train_images = torch.tensor(train_dataset["image"], dtype=torch.uint8)
  train_labels = torch.tensor(train_dataset["label"], dtype=torch.int64)

  test_images = torch.tensor(test_dataset["image"], dtype=torch.uint8)
  test_labels = torch.tensor(test_dataset["label"], dtype=torch.int64)

  # Images pixels are grayscale 0-255, normalize them to 0-1.
  normalized_train_images = train_images / 255.0
  normalized_test_images = test_images / 255.0

  # Flatten the images to 1D tensors
  normalized_train_images = normalized_train_images.view(-1, 28 * 28)
  normalized_test_images = normalized_test_images.view(-1, 28 * 28)

  # Expand train_labels to one-hot encoding ([1, 2, 3] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  train_labels = torch.nn.functional.one_hot(train_labels, num_classes=10).float()
  test_labels = torch.nn.functional.one_hot(test_labels, num_classes=10).float()

  return normalized_train_images, train_labels, normalized_test_images, test_labels

def load_datasets_as_lists():
  # Load datasets as tensors
  train_images, train_labels, test_images, test_labels = load_datasets()

  # Convert tensors to lists
  return train_images.tolist(), train_labels.tolist(), test_images.tolist(), test_labels.tolist()
