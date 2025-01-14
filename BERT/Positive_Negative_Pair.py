import pandas as pd
import json
import numpy as np

# Load the data
df_train = pd.read_csv('/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/train.csv', on_bad_lines='skip')
df_corpus = pd.read_csv('/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/preprocessed_corpus.csv')

# Load predictions file
with open('/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/predict_most_dissimilar.json', 'r') as f:
    predictions = json.load(f)

training_data = []

for pred in predictions:
    qid = pred['query_id']
    neg_cid = pred['candidates']['doc_ids'][0]  # Get negative example cid
    
    # Get query text from train.csv
    query_row = df_train[df_train['qid'] == qid]
    if query_row.empty:
        continue
    query_text = query_row['question'].iloc[0]
    
    # Get positive example
    ground_truth_str = query_row['cid'].iloc[0]
    ground_truth_cids = ground_truth_str.strip('[]').split()
    pos_cid = ground_truth_cids[0]  # Take first positive example
    
    # Get texts from corpus
    pos_matches = df_corpus.loc[df_corpus['cid'] == int(pos_cid), 'text']
    neg_matches = df_corpus.loc[df_corpus['cid'] == int(neg_cid), 'text']
    
    if pos_matches.empty or neg_matches.empty:
        continue  # Skip if no matching text found
    
    pos_text = pos_matches.iloc[0]
    neg_text = neg_matches.iloc[0]
    
    # Create entry in required format
    entry = {
        "query": query_text,
        "pos": [pos_text],
        "neg": [neg_text]
    }
    training_data.append(entry)

# Save to json file
with open('fine_tune_training.json', 'w', encoding='utf-8') as f:
    for entry in training_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

