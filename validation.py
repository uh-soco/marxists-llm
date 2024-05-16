
from transformers import pipeline

import torch

import random

model_name = "openai-community/gpt2"

prompts = open("prompts.txt").readlines()

models = [model_name, './my_fine_tuned_model-causal/']

for prompt in prompts:

    print( prompt )

    for model in models:

        torch.manual_seed(0)
        random.seed(0)
        
        generator = pipeline('text-generation', model = model)
        text = generator( prompt ) 

        print( model, text[0]['generated_text'].replace('\n', ' ') )

    print()