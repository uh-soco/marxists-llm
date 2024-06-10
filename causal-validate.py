
from transformers import pipeline

import torch

import random

models = """openai-community/gpt2
distilbert/distilgpt2
chavinlo/alpaca-native
bigscience/bloom
microsoft/phi-2"""
models = models.split("\n")

prompts = open("prompts.txt").readlines()

for model_name in models:

    for prompt in prompts:

        torch.manual_seed(0)
        random.seed(0)

        print( prompt )
        generator = pipeline('text-generation', model = f"./{model_name.replace('/', '_')}-finetuned-causal"))

        for i in range( 10 ):
            text = generator( prompt ) 
            print( model_name, text[0]['generated_text'].replace('\n', ' ') )

        print()