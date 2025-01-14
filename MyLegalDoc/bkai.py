import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load preprocessed corpus
df = pd.read_csv('/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/preprocessed_corpus.csv')

# Initialize the model
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
model = model.to(device)

def encode(lst = [], convert_to_tensor=True, batch_size=128):
    vectors = []
    # Create progress bar
    with tqdm(total(len(lst), desc="Encoding texts")) as pbar:
        # Process in batches
        for i in range(0, len(lst), batch_size):
            batch = lst[i:i + batch_size]
            encoded_batch = model.encode(batch, convert_to_tensor=True, device=device)
            # Move to CPU before converting to numpy
            if torch.cuda.is_available():
                encoded_batch = encoded_batch.cpu()
            vectors.extend([np.array(arr) for arr in encoded_batch.numpy()])
            pbar.update(len(batch))
    return vectors

# Encode the text column
df['vector'] = encode(lst=list(df['text']))

# Save the encoded corpus
df.to_json('encoded_corpus.json')
