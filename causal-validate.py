import random
import csv

from transformers import pipeline
import torch

models = """openai-community/gpt2
distilbert/distilgpt2"""
models = models.split("\n")

out = csv.writer( open("causal.csv", "w") )
out.writerow(  ["model", "prompt", "output"] )

prompts = open("prompts.txt").readlines()

for model_name in models:

    for prompt in prompts:

        torch.manual_seed(0)
        random.seed(0)

        generator = pipeline('text-generation', model = f"./models/{model_name.replace('/', '_')}-finetuned-causal/")

        for i in range( 10 ):
            text = generator( prompt )
            out.writerow(  [model_name, prompt.strip(), text[0]['generated_text'].replace('\n', ' ').strip() ] )
