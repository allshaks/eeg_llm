from datasets import load_dataset
from transformers import GPT2Tokenizer 
from transformers import GPT2Config, GPT2LMHeadModel 
from transformers import Trainer, TrainingArguments 
from transformers import pipeline 

# load the tiny Shakespeare dataset 
dataset = load_dataset("tiny_shakespeare")

texts = dataset['train']['text']

# tokenize the data 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

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

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True, 
    num_train_epochs=3, 
    per_device_eval_batch_size=4, 
    save_steps=500, 
    save_total_limit=2, 
    logging_dir="./logs",
    logging_steps=100,
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_datasets['train'],
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)


generated_text = text_gen("To be or not to be", max_length=50, num_return_sequences=2)
print(generated_text[0]['generated_text'])