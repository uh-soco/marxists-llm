import torch
import pickle
import pandas

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM

def embedding( model_name , output_name ):

    word_embeddings = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    for token in tokenizer.vocab.keys():
        input = tokenizer(token, return_tensors="pt")
        input_id = input["input_ids"]

        with torch.no_grad():
            outputs = model.base_model(**input)
            last_hidden_state = outputs.last_hidden_state

        token_embeddings = last_hidden_state[0]
        tokens = tokenizer.convert_ids_to_tokens(input_id[0])

        for token, embedding in zip(tokens, token_embeddings):
            word_embeddings[token] = embedding.numpy()

    df = pandas.DataFrame(word_embeddings).T
    df.to_csv( output_name )


embedding("../models/microsoft_deberta-v3-base-finetuned-masked-model", "marxist-embedding.csv")
embedding("microsoft/deberta-v3-base", "capitalist-embedding.csv")
