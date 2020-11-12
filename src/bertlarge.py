
import torch
import sys
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)



  

