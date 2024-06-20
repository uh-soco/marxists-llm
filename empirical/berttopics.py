import random
import torch

import re

import pandas

from sentence_transformers import SentenceTransformer, models
from bertopic import BERTopic

data = pandas.read_csv("realDonaldTrump_in_office.csv")

## remove links
data['Tweet.Text'] = data['Tweet.Text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

for model_name in ["microsoft/deberta-v3-base", "../models/microsoft_deberta-v3-base-finetuned-masked-model/"]:

    torch.manual_seed(0)
    random.seed(0)

    word_embedding_model = models.Transformer(model_name, max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    transformer_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    topic_model = BERTopic(embedding_model=transformer_model)

    print( model_name )
    topics, probs = topic_model.fit_transform(data['Tweet.Text'])
    print( topic_model.get_topic_info() )