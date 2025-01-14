from datasets import Dataset, load_dataset
import os
! pip install -U accelerate
! pip install -U transformers
os.environ["WANDB_DISABLED"] = "true"
import json
from sentence_transformers import (
    SentenceTransformer,
    losses,
    InputExample
)
from torch.utils.data import DataLoader

# 1. Load the model
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

# 2. Load and prepare the dataset
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Prepare training examples using InputExample
    train_examples = []
    for item in data:
        query = item['query']
        pos = item['pos'][0]  # Taking first positive example
        
        # Create InputExample for positive pair
        train_examples.append(InputExample(
            texts=[query, pos],
            label=1.0  # Positive pair
        ))
        
        # Create InputExample for negative pair
        for neg in item['neg']:
            train_examples.append(InputExample(
                texts=[query, neg],
                label=0.0  # Negative pair
            ))
    
    return train_examples

# Load the data
train_examples = load_json_data('/VeMienTay/fine_tune_training.json')

# 3. Create train dataloader
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=48
)

# 4. Define the loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# 5. Configure training parameters
warmup_steps = int(len(train_dataloader) * 0.1)  # 10% of training data for warmup

# 6. Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': 2e-5},
    output_path='/VeMienTay/finetuned',
    save_best_model=True,
    show_progress_bar=True,
    checkpoint_path='/VeMienTay/finetuned/checkpoints',
    checkpoint_save_steps=7500,  # Save every 7500 steps
    checkpoint_save_total_limit=2  # Keep only the 2 most recent checkpoints
)

# 7. Save the final model
model.save('/VeMienTay/final') 