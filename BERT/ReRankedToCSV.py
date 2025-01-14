import pandas as pd

# Read the reranked JSON file
reranked_df = pd.read_json('/home/LegalDocumentRetrieval-20241027T111633Z-001/BERT/predict_top50_reranked.json')

# Open output file
with open('predict_top50.txt', 'w') as f:
    # Process each row
    for _, row in reranked_df.iterrows():
        qid = row['query_id']
        doc_ids = row['candidates']['doc_ids']
        scores = row['candidates']['scores']
        
        # Create pairs of (doc_id, score) and sort by score descending
        doc_score_pairs = list(zip(doc_ids, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 10 doc_ids after sorting
        top_10_docs = [pair[0] for pair in doc_score_pairs[:10]]
        
        # Format line: qid cid1 cid2 ... cid10
        line = f"{qid} {' '.join(map(str, top_10_docs))}\n"
        f.write(line)