import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load data with both question and qid columns
df = pd.read_csv('/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/public_test.csv')

# Determine the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the model with the specified device 
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)

def encode(lst = [], convert_to_tensor=True, batch_size=128):
    vectors = []
    with tqdm(total=len(lst), desc="Encoding questions") as pbar:
        for i in range(0, len(lst), batch_size):
            batch = lst[i:i + batch_size]
            encoded_batch = model.encode(batch, convert_to_tensor=True)
            if device == 'cuda':
                encoded_batch = encoded_batch.cpu()
            vectors.extend([np.array(arr) for arr in encoded_batch.numpy()])
            pbar.update(len(batch))
    return vectors

# Encode questions while preserving original columns
df['question_vector'] = encode(lst=list(df['question']))

# Select only the required columns
output_df = df[['question', 'qid', 'question_vector']]

# Save to JSON
output_df.to_json('encoded_public_test.json')