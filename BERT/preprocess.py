import pandas as pd

class ProcessCorpus:
    def __init__(self, filepath='/home/LegalDocumentRetrieval-20241027T111633Z-001/LegalDocumentRetrieval/corpus.csv'):
        self.df = pd.read_csv(filepath, usecols=['text', 'cid'])

    def process_text(self):
        self.df['text'] = self.df['text'].apply(lambda x: ' '.join(x.split()))
        self.save_to_csv()

    def save_to_csv(self):
        output_path = 'preprocessed_corpus.csv'
        self.df.to_csv(output_path, index=False)


if __name__ == "__main__":
    processor = ProcessCorpus()
    processor.process_text()