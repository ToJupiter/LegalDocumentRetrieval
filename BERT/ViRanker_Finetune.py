import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW  # Changed from transformers.AdamW

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('namdp-ptit/ViRanker')
model = AutoModelForSequenceClassification.from_pretrained('namdp-ptit/ViRanker')

# Load your training data from the JSON file
with open('/kaggle/input/negativepairing/fine_tune_training.json', 'r', encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f]

# Create training pairs (query, document) with labels (1 for positive, 0 for negative)
train_pairs = []
for item in train_data:
    query = item['query']
    for pos_doc in item['pos']:
        train_pairs.append((query, pos_doc, 1))
    for neg_doc in item['neg']:
        train_pairs.append((query, neg_doc, 0))

class RankerDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, doc, label = self.pairs[idx]
        encoding = self.tokenizer(query, doc, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=self.max_length, 
                                return_tensors='pt')
        
        # Convert label to float
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)  # Changed to float
        }

train_dataset = RankerDataset(train_pairs, tokenizer)

batch_size = 8

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 3

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

model.save_pretrained('fine_tuned_viranker')
tokenizer.save_pretrained('fine_tuned_viranker')