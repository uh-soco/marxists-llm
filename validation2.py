from transformers import pipeline

import torch

import random

prompts = open("prompts.txt").readlines()

models = ['distilbert/distilroberta-base', './my_fine_tuned_model-masked/']

for prompt in prompts:

    print( prompt )

    for model in models:

        torch.manual_seed(0)
        random.seed(0)
        
        generator = pipeline('fill-mask', model = model )
        text = generator(prompt.lower() + '<mask>.' ) 

        print( model, text[0]['sequence'] )

    print()