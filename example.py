from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

model_name = "distilbert/distilroberta-base"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

datasets = load_dataset("text", data_files={"train": './data/*.txt', "validation": './data/*.txt'})

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = 2**6

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


from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name)


training_args = TrainingArguments(
    evaluation_strategy = "epoch",
    learning_rate=2e-10,
    weight_decay=0.01,
    warmup_steps=500,
    output_dir="./results",
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# Train the model
trainer.train()

# Save fine-tuned model and tokenizer
trainer.save_model("./my_fine_tuned_model-causal")
tokenizer.save_pretrained("./my_fine_tuned_model-causal")
