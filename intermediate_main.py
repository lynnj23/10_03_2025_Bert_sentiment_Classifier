from datasets import load_dataset
from transformers import AutoTokenizer

# Define the dataset splits
split_names = ["train", "test"]

# Load the IMDB dataset for both splits.
imdb_datasets = load_dataset("imdb", split=split_names)

# Manually assign each split to a variable.
train_dataset = imdb_datasets[0]
test_dataset = imdb_datasets[1]

# Optionally, create a dictionary for easier access later.
datasets = {
    "train": train_dataset,
    "test": test_dataset
}

# Print the dictionary containing both datasets.
print(datasets)
