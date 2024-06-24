import transformers
import csv

from transformers import pipeline
import torch

models = """openai-community/gpt2"""
models = models.split("\n")

out = csv.writer( open("outputs/causal-compare.csv", "w") )
out.writerow(  ["model", "prompt", "output"] )

prompts = open("prompts.txt").readlines()

for model_name in models:

    for prompt in prompts:

        transformers.enable_full_determinism( 0 )

        generator = pipeline('text-generation', model = f"./models/{model_name.replace('/', '_')}-finetuned-causal-model/")

        for i in range( 2 ):
            text = generator( prompt )
            out.writerow(  ["Marxist LLM", prompt.strip(), text[0]['generated_text'].replace('\n', ' ').strip() ] )

        transformers.enable_full_determinism( 0 )

        generator = pipeline('text-generation', model = model_name )

        for i in range( 2 ):
            text = generator( prompt )
            out.writerow(  ["Capitalist LLM", prompt.strip(), text[0]['generated_text'].replace('\n', ' ').strip() ] )
