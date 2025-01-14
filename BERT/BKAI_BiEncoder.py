import torch
import torch.nn.functional as F
import pandas as pd
import zipfile
import json

# ...existing code...

# Load JSON files using Pandas
train_df = pd.read_json('/kaggle/input/encoded-public-test/encoded_public_test.json')
# train_df = train_df.head(1000)
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

# Compute cosine similarity using PyTorch on CUDA
similarity_matrix = torch.matmul(train_vectors, corpus_vectors.T)

# Get top K similar documents
top_k = 50  # Adjust as needed
top_k_values, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)

# Create results for both TXT and JSON formats
json_results = []

with open('predict_top50.txt', 'w') as f:
    for i, (indices, scores) in enumerate(zip(top_k_indices, top_k_values)):
        qid = train_ids[i]
        
        # Convert indices and scores to Python lists
        top_cids = [str(corpus_ids[idx.item()]) for idx in indices]
        similarity_scores = [score.item() for score in scores]
        
        # Write to TXT format
        line = f"{qid} {' '.join(top_cids)}\n"
        f.write(line)
        
        # Prepare JSON format
        # Format suitable for re-ranker: including query_id, candidate_ids, and their scores
        json_entry = {
            "query_id": qid,
            "candidates": {
                "doc_ids": top_cids,
                "scores": similarity_scores,
                # Additional fields that might be useful for re-ranker:
            }
        }
        json_results.append(json_entry)

# Save JSON results
with open('predict_top50.json', 'w', encoding='utf-8') as f:
    json.dump(json_results, f, ensure_ascii=False, indent=2)

# Zip both files
with zipfile.ZipFile('predict_top50.zip', 'w') as zipf:
    zipf.write('predict_top50.txt')
    zipf.write('predict_top50.json')

# ...existing code...
