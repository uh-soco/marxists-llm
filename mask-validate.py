import random
import csv

from transformers import pipeline
import torch

models = """google-bert/bert-base-cased
distilbert/distilbert-base-cased
FacebookAI/roberta-base
microsoft/deberta-v3-base
distilbert/distilroberta-base"""
models = models.split("\n")

prompts = open("prompts.txt").readlines()

out = csv.writer( open("masked.csv", "w") )
out.writerow(  ["model", "prompt", "output"] )

for model_name in models:

    for prompt in prompts:

        torch.manual_seed(0)
        random.seed(0)

        generator = pipeline('text-generation', model = f"./{model_name.replace('/', '_')}-finetuned-masked")

        for i in range( 10 ):
            text = generator( prompt + '[MASK]' ) 
            out.writerow(  [model_name, prompt.strip(), text[0]['sequence'].strip() ] )
