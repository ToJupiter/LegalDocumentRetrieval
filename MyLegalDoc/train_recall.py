import pandas as pd
import numpy as np

df_train = pd.read_csv(
    '/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/train.csv',
    on_bad_lines='skip'  # Skip lines with unexpected number of fields
)
df_train = df_train.head(1000)

# Load the data

count_retrieve = np.zeros(10)  # Initialize the count array

with open("/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/predict_train_mini.txt", "r") as f:
    for line in f:
        line = line.strip().split()
        qid = line[0]
        retrieved_cids = line[1:]  # cid_1 to cid_10

        # Get the ground truth cids for this qid from the DataFrame
        ground_truth_row = df_train[df_train['qid'] == int(qid)]
        if not ground_truth_row.empty:
            ground_truth_str = ground_truth_row['cid'].iloc[0]
            # Convert "[31682 31677]" format to list of strings
            ground_truth_cids = ground_truth_str.strip('[]').split()
            ground_truth_cids = [str(cid) for cid in ground_truth_cids]

        # Check each ground truth cid against the retrieved cids
        for true_cid in ground_truth_cids:
            for i, retrieved_cid in enumerate(retrieved_cids):
                if true_cid == retrieved_cid:
                    count_retrieve[i] += 1
                    break  # Move to the next true_cid once it's found

print("Retrieval Rank Counts:", count_retrieve)

# Calculate recall@k
recall_at_1 = count_retrieve[0] / len(df_train)
recall_at_5 = np.sum(count_retrieve[:5]) / len(df_train)
recall_at_10 = np.sum(count_retrieve) / len(df_train)

print(f"Recall@1: {recall_at_1:.4f}")
print(f"Recall@5: {recall_at_5:.4f}")
print(f"Recall@10: {recall_at_10:.4f}")

