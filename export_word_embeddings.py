import transformers
import tokenizers
import random
import torch

import csv

torch.manual_seed(0)
random.seed(0)

from datasets import load_dataset
from datasets import ClassLabel

from transformers import AutoTokenizer
from transformers import AutoModel

model_name = "distilbert/distilbert-base-uncased"
model_name = "./my_fine_tuned_model-masked/"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_word_embedding(word):
    # Create input sentence with special tokens
    input_text = f"[CLS] {word} [SEP]"
    input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

    # Get hidden states from BERT
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.last_hidden_state

    # Average hidden states for the word token
    word_embedding = hidden_states.mean(dim=1).squeeze()

    return word_embedding.numpy()

csvout = csv.writer( open('model.csv', 'w', encoding='UTF8', newline='') )

for word in tokenizer.vocab.keys():
    embedding = get_word_embedding(word)
    csvout.writerow( [ word ] + embedding.tolist() )