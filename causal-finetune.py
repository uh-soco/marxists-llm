import os

import transformers
import torch

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import gc

PATH = os.environ.get("PATH", "./")

models = """openai-community/gpt2
distilbert/distilgpt2
microsoft/phi-2
meta-llama/Llama-3.2-3B
mistralai/Mixtral-8x7B-v0.1
google/gemma-3-4b-it
google/gemma-3-12b-it
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
openai/gpt-oss-20b
meta-llama/Llama-3.3-70B-Instruct
"""
models = models.split("\n")

for model_name in models:

    print(f"Start {model_name}")
    try:
        transformers.enable_full_determinism( 0 )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token

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
        
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        training_args = TrainingArguments(
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=10,
            weight_decay=0.01,
            output_dir=  f"{PATH}/models/{model_name.replace('/', '_')}-finetuned-causal-output",
            logging_dir=  f"{PATH}/models/{model_name.replace('/', '_')}-finetuned-causal-logs",
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
        trainer.save_model( f"{PATH}/models/{model_name.replace('/', '_')}-finetuned-causal")
        tokenizer.save_pretrained(f"{PATH}/models/{model_name.replace('/', '_')}-finetuned-causal")
    except Exception as e:
        print( f"Failed {model_name}" )
        print( e )

    finally:
        # Always clean up, even if training failed
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
