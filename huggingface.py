from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from sklearn.model_selection import train_test_split


# Load the tiny Shakespeare dataset
dataset = load_dataset("tiny_shakespeare")

# Tokenize the data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, return_special_tokens_mask=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# split the data into train and test data 
train_test_split = tokenized_datasets['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

print(f"{len(train_dataset)}")
'''
# Ensure labels are included in the dataset
def add_labels(batch):
    batch["labels"] = batch["input_ids"].copy()
    return batch

tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

# Configure and initialize the model
gpt_config = GPT2Config(
    n_embd=512,
    n_layer=6,
    n_head=8,
    n_positions=512,
    resid_dropout=0.1,
    attn_pdrop=0.1,
)

model = GPT2LMHeadModel(gpt_config)
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Train the model
trainer.train()
'''