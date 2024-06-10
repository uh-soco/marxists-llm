
from transformers import pipeline

import torch

import random

models = """google-bert/bert-base-cased
distilbert/distilbert-base-cased
FacebookAI/roberta-base
microsoft/deberta-v3-base
distilbert/distilroberta-base"""
models = models.split("\n")

prompts = open("prompts.txt").readlines()

for model_name in models:

    for prompt in prompts:

        torch.manual_seed(0)
        random.seed(0)

        print( prompt )
        generator = pipeline('text-generation', model = f"./{model_name.replace('/', '_')}-finetuned-masked"))

        for i in range( 10 ):
            text = generator( prompt + '[MASK]' ) 
            print( model_name, text[0]['generated_text'].replace('\n', ' ') )

        print()