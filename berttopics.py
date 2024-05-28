import random
import torch

from sentence_transformers import SentenceTransformer, models
from bertopic import BERTopic

from sklearn.datasets import fetch_20newsgroups
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']


for model in ["distilbert/distilbert-base-uncased", "./my_fine_tuned_model-masked/"]:

    torch.manual_seed(0)
    random.seed(0)

    word_embedding_model = models.Transformer(model, max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    transformer_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    topic_model = BERTopic(embedding_model=transformer_model)

    print( model )
    topics, probs = topic_model.fit_transform(docs)
    print( topic_model.get_topic_info() )