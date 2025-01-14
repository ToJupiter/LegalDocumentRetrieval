import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv('/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/train.csv')

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize the model
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
model = model.to(device)

def encode(lst = [], convert_to_tensor=True, batch_size=128):
    vectors = []
    with tqdm(total=len(lst), desc="Encoding questions") as pbar:
        for i in range(0, len(lst), batch_size):
            batch = lst[i:i + batch_size]
            encoded_batch = model.encode(batch, convert_to_tensor=True, device=device)
            if torch.cuda.is_available():
                encoded_batch = encoded_batch.cpu()
            vectors.extend([np.array(arr) for arr in encoded_batch.numpy()])
            pbar.update(len(batch))
    return vectors

# Encode questions for both train and test sets
train_df['question_vector'] = encode(lst=list(train_df['question']))
test_df['question_vector'] = encode(lst=list(test_df['question']))

# Save encoded datasets
train_df.to_json('encoded_train.json')
test_df.to_json('encoded_test.json')

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

