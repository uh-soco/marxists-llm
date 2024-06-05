from transformers import pipeline

import torch

import random

prompts = open("prompts.txt").readlines()

# model_checkpoint = "distilbert/distilroberta-base"
# model_checkpoint = "FacebookAI/roberta-base"
model_checkpoint = "distilbert/distilbert-base-uncased"

models = [model_checkpoint, './my_fine_tuned_model-masked/']
# models = ["google-bert/bert-base-cased", './my_fine_tuned_model-masked/']

for prompt in prompts:

    print( prompt )

    for model in models:

        torch.manual_seed(0)
        random.seed(0)
        
        generator = pipeline('fill-mask', model = model )
        text = generator(prompt.lower() + '[MASK].' ) # '<mask>' )

        print( model, text[0]['sequence'] )

    print()