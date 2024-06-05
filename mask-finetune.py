import transformers
import tokenizers
import random
import torch

torch.manual_seed(0)
random.seed(0)

from datasets import load_dataset
from datasets import ClassLabel

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments

datasets = load_dataset("text", data_files={"train": './data/*.txt', "validation": './data/*.txt'})

# model_checkpoint = "FacebookAI/roberta-base"
# model_checkpoint = "distilbert/distilroberta-base"
# model_checkpoint = "google-bert/bert-base-cased"
model_checkpoint = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer.normalizer = tokenizers.normalizers.BertNormalizer()
# tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = 2*4

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

model = AutoModelForMaskedLM.from_pretrained( model_checkpoint )

training_args = TrainingArguments(
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    num_train_epochs=8,
    weight_decay=0.01,
    output_dir="./results",
    logging_dir='./logs',
    logging_steps=10,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.save_model("./my_fine_tuned_model-masked")
# tokenizer.save_pretrained("./my_fine_tuned_model-masked")