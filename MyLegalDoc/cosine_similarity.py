import torch
import torch.nn.functional as F
import pandas as pd
import zipfile

# ...existing code...

# Load JSON files using Pandas
train_df = pd.read_json('encoded_public_test.json')
corpus_df = pd.read_json('encoded_corpus.json')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# Extract IDs and vectors
train_ids = train_df['qid'].tolist()
train_vectors = torch.tensor(train_df['question_vector'].tolist(), dtype=torch.float32).to(device)
corpus_ids = corpus_df['cid'].tolist()
corpus_vectors = torch.tensor(corpus_df['vector'].tolist(), dtype=torch.float32).to(device)

# Normalize vectors
train_vectors = F.normalize(train_vectors, p=2, dim=1)
corpus_vectors = F.normalize(corpus_vectors, p=2, dim=1)

# Compute cosine similarity using PyTorch on CUDA
similarity_matrix = torch.matmul(train_vectors, corpus_vectors.T)

# Get top K similar documents
top_k = 10  # Adjust as needed
top_k_values, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)

# Write results to predict.txt in required format
with open('predict.txt', 'w') as f:
    for i, indices in enumerate(top_k_indices):
        qid = train_ids[i]
        top_cids = [str(corpus_ids[idx.item()]) for idx in indices]
        line = f"{qid} {' '.join(top_cids)}\n"
        f.write(line)

# Zip the predict.txt file
with zipfile.ZipFile('predict.zip', 'w') as zipf:
    zipf.write('predict.txt')

# ...existing code...
