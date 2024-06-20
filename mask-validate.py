import random
import csv

import transformers
import torch

from transformers import pipeline

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

        transformers.enable_full_determinism( 0 )
        generator = pipeline('fill-mask', model = f"./models/{model_name.replace('/', '_')}-finetuned-masked-model/")

        try:
            text = generator( mask.split('|')[1] )
        except:
            text = generator( mask.split('|')[1].replace('[MASK]', '<mask>') )
            
        for i in range( min( 5 , len(text) ) ):
            out.writerow(  [f"finetuned-{model_name}", mask, text[i]['token_str'].strip() ] )

    for mask in masks:

        transformers.enable_full_determinism( 0 )
        generator = pipeline('fill-mask', model = model_name )

        try:
            text = generator( mask.split('|')[1] )
        except:
            text = generator( mask.split('|')[1].replace('[MASK]', '<mask>') )

        for i in range( min( 5, len(text) ) ):
            out.writerow(  [model_name, mask, text[i]['token_str'].strip() ] )
