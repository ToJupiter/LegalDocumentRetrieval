import torch
import torch.nn.functional as F
import pandas as pd
import zipfile
import json

# ...existing code...

# Load JSON files using Pandas
train_df = pd.read_json('/kaggle/input/encodedtrainfull/encoded_train_full.json')
corpus_df = pd.read_json('/kaggle/input/encoded/encoded_corpus.json')

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

# Get most dissimilar document
json_results = []
batch_size = 128 # Process 8 QIDs at a time

with open('predict_most_dissimilar.txt', 'w') as f:
    for i in range(0, len(train_ids), batch_size): 
        # Process in batches
        batch_train_vectors = train_vectors[i:i+batch_size]
        batch_train_ids = train_ids[i:i+batch_size]
        
        # Compute cosine similarity using PyTorch on CUDA for the batch
        similarity_matrix = torch.matmul(batch_train_vectors, corpus_vectors.T)
        
        # Get most dissimilar document (smallest similarity score)
        min_values, min_indices = torch.min(similarity_matrix, dim=1)
        
        for j, (idx, score) in enumerate(zip(min_indices, min_values)):
            qid = batch_train_ids[j]
            cid = str(corpus_ids[idx.item()])
            
            # Write to TXT format
            f.write(f"{qid} {cid}\n")

            # Prepare JSON format
            json_entry = {
                "query_id": qid,
                "candidates": {
                    "doc_ids": [cid],
                    "scores": [score.item()],
                }
            }
            json_results.append(json_entry)

# Save JSON results
with open('predict_most_dissimilar.json', 'w', encoding='utf-8') as f:
    json.dump(json_results, f, ensure_ascii=False, indent=2)

# Zip both files
with zipfile.ZipFile('predict_most_dissimilar.zip', 'w') as zipf:
    zipf.write('predict_most_dissimilar.txt')
    zipf.write('predict_most_dissimilar.json')

# ...existing code...