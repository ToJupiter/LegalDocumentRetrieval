# Use a pipeline as a high-level helper
from transformers import pipeline
import pandas as pd

pipe = pipeline("fill-mask", model="vinai/phobert-large")

# Load corpus.csv
corpus = pd.read_csv('corpus.csv')

# Mask words in the corpus
masked_corpus = [pipe(text) for text in corpus['text']]

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-large")

# Tokenize Masked Corpus
encoded_inputs = tokenizer(masked_corpus, return_tensors='pt', padding=True)

# Run the Model
outputs = model(**encoded_inputs)