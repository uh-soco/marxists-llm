import random
import torch

from sentence_transformers import SentenceTransformer, models
from bertopic import BERTopic

word_embedding_model = models.Transformer("distilbert/distilbert-base-uncased", max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

topic_model_normal = BERTopic(embedding_model=model)

word_embedding_model = models.Transformer("./my_fine_tuned_model-masked/", max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

topic_model_marx = BERTopic(embedding_model=model)

from sklearn.datasets import fetch_20newsgroups
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

torch.manual_seed(0)
random.seed(0)

topics, probs = topic_model_normal.fit_transform(docs)
print( topic_model_normal.get_topic_info() )

torch.manual_seed(0)
random.seed(0)

topics, probs = topic_model_marx.fit_transform(docs)
print( topic_model_marx.get_topic_info() )