import transformers
import tokenizers
import random
import torch

from datasets import load_dataset
from datasets import ClassLabel

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

models = """google-bert/bert-base-cased
distilbert/distilbert-base-cased
FacebookAI/roberta-base
microsoft/deberta-v3-base
distilbert/distilroberta-base"""
models = models.split("\n")

for model_name in models:

    print(f"Start {model_name}")
    torch.manual_seed(0)
    random.seed(0)

    datasets = load_dataset("text", data_files={"train": './data/*.txt', "validation": './data/*.txt'})

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.normalizer = tokenizers.normalizers.BertNormalizer( lowercase=False )
  
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    block_size = 2*8

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

    model = AutoModelForMaskedLM.from_pretrained( model_name )

    training_args = TrainingArguments(
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        output_dir=  f"./models/{model_name.replace('/', '_')}-finetuned-masked-output",
        logging_dir= f"./models/{model_name.replace('/', '_')}-finetuned-masked-logs",
        logging_steps=10,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    #import math
    #eval_results = trainer.evaluate()
    #print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print( f"Saving {model_name}" )
    trainer.save_model(f"./models/{model_name.replace('/', '_')}-finetuned-masked-model")
    tokenizer.save_pretrained(f"./models/{model_name.replace('/', '_')}-finetuned-masked-model")
