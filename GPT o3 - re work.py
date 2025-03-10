# Import the datasets and transformers packages
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    Trainer, TrainingArguments
import numpy as np
import pandas as pd

# Load the train and test splits of the imdb dataset
splits = ["train", "test"]
# Use a different name for the loop variable to avoid confusion
ds = {split: dataset for split, dataset in zip(splits, load_dataset("imdb", split=splits))}

# Thin out the dataset to make it run faster for this example
for split in splits:
    ds[split] = ds[split].shuffle(seed=42).select(range(500))

# Show the dataset
print(ds)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    """Preprocess the imdb dataset by returning tokenized examples."""
    return tokenizer(examples["text"], truncation=True, padding="max_length")


# Tokenize the datasets
tokenized_ds = {}
for split in splits:
    tokenized_ds[split] = ds[split].map(preprocess_function, batched=True)

# Check that we tokenized the examples properly.
# Note: This assertion assumes that the first 5 token ids of the first training example match these values.
expected_tokens = [101, 2045, 2003, 2053, 7189]
first_example_tokens = tokenized_ds["train"][0]["input_ids"][:5]
assert first_example_tokens == expected_tokens, f"Expected {expected_tokens} but got {first_example_tokens}"

# Show the first example of the tokenized training set
print(tokenized_ds["train"][0]["input_ids"])

# Load the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},  # For converting predictions to strings
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

# Freeze all the parameters of the base model.
# For DistilBERT, the underlying model is stored in the attribute 'distilbert', not 'base_model'.
for param in model.distilbert.parameters():
    param.requires_grad = False

# (Optional) You can print the classifier component if needed:
print(model.classifier)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# Set up the Trainer with training and evaluation configurations.
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/sentiment_analysis",
        learning_rate=2e-3,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Convert the test dataset to a pandas DataFrame
df = pd.DataFrame(tokenized_ds["test"])
# Keep only the 'text' and 'label' columns
df = df[["text", "label"]]

# Replace <br /> tags in the text with spaces
df["text"] = df["text"].str.replace("<br />", " ")

# Add the model predictions to the DataFrame.
predictions = trainer.predict(tokenized_ds["test"])
# predictions[0] contains the logits. We take argmax to get the predicted label.
df["predicted_label"] = np.argmax(predictions[0], axis=1)

# Show the first two rows of the DataFrame
print(df.head(2))
