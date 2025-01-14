import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

# Load all required data
predictions = pd.read_json('/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/predict_top50.json')
test_df = pd.read_csv('/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/public_test.csv')
corpus_df = pd.read_csv('/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/corpus.csv')

# Debugging: Print first few rows to verify data format
print("First few corpus rows:", corpus_df.head())
print("First few predictions:", predictions.head())

# Fix: Create corpus dictionary with correct column mapping
corpus_dict = dict(zip(corpus_df['text'], corpus_df['text']))

# Setup model and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('namdp-ptit/ViRanker')
model = AutoModelForSequenceClassification.from_pretrained('namdp-ptit/ViRanker').to(device)
model.eval()

reranked_results = []

# Process each query
for _, row in predictions.iterrows():
    # Fix: Use correct key from predictions json
    qid = row['query_id']  # or 'query_id' depending on actual json format
    query_text = test_df[test_df['qid'] == qid]['question'].iloc[0]
    doc_ids = row['candidates']['doc_ids']
    
    # Debug print
    print(f"Processing qid: {qid}")
    print(f"First few doc_ids: {doc_ids[:3]}")
    
    # Create pairs with proper error handling
    pairs = []
    for doc_id in doc_ids:
        try:
            # Convert doc_id to string if it's numeric
            doc_id_str = str(doc_id)
            pairs.append([query_text, corpus_dict[doc_id_str]])
        except KeyError:
            print(f"Warning: doc_id {doc_id} not found in corpus")
            # Skip this pair or handle missing documents as needed
    
    # Re-rank in batches
    batch_size = 8  # Smaller batch size due to longer texts
    all_scores = []
    
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = tokenizer(batch_pairs, padding=True, truncation=True, 
                             return_tensors='pt', max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            scores = model(**inputs, return_dict=True).logits.view(-1,).float()
            all_scores.extend(scores.cpu().numpy().tolist())
    
    # Create result entry
    json_entry = {
        "query_id": qid,
        "candidates": {
            "doc_ids": doc_ids,
            "scores": all_scores
        }
    }
    reranked_results.append(json_entry)

# Save results
with open('predict_top50_reranked.json', 'w') as f:
    json.dump(reranked_results, f, indent=2)
