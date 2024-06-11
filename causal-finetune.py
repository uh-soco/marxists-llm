from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import random
import torch

models = """openai-community/gpt2
distilbert/distilgpt2
chavinlo/alpaca-native
bigscience/bloom
microsoft/phi-2"""
models = models.split("\n")

for model_name in models:

    print(f"Start {model_name}")
    torch.manual_seed(0)
    random.seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.normalizer = tokenizer.normalizers.BertNormalizer( lowercase=False )

    datasets = load_dataset("text", data_files={"train": './data/*.txt', "validation": './data/*.txt'})

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    block_size = 2**7

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
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
    
    model = AutoModelForCausalLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        output_dir= f"./models/{model_name.replace('/', '_')}-finetuned-causal-output",
        logging_dir= f"./models/{model_name.replace('/', '_')}-finetuned-causal-logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )

    trainer.train()
    print( f"Saving {model_name}" )
    trainer.save_model(f"./models/{model_name.replace('/', '_')}-finetuned-causal")
    tokenizer.save_pretrained(f"./models/{model_name.replace('/', '_')}-finetuned-causal")
