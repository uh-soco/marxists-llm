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

masks = open("masks.txt").readlines()

out = csv.writer( open("masked.csv", "w") )
out.writerow(  ["model", "prompt", "output"] )

for model_name in models:

    for mask in masks:

        torch.manual_seed(0)
        random.seed(0)

        generator = pipeline('fill-mask', model = f"./models/{model_name.replace('/', '_')}-finetuned-masked-model/")

        text = generator( mask.split('|')[1] )
        for i in range( min( 10 , len(text) ) ):
            out.writerow(  [model_name, mask, text[i]['token_str'].strip() ] )

    for mask in masks:

        torch.manual_seed(0)
        random.seed(0)

        generator = pipeline('fill-mask', model = model_name )

        text = generator( mask.split('|')[1] )
        for i in range( min( 10 , len(text) ) ):
            out.writerow(  [model_name, mask, text[i]['token_str'].strip() ] )
