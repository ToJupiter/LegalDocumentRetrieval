import pandas as pd
import jsonlines
import json

with open('/home/LegalDocumentRetrieval-20241027T111633Z-001/BERT/fine_tune_training.json', 'r') as f:
    json_data = json.load(f)

with open('fine_tune_training.jsonl', 'w') as jsonl_output:
    for entry in json_data:
        json.dump(entry, jsonl_output)
        jsonl_output.write('\n')