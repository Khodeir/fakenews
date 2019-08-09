from reading.build_vocab import load_vocab, load_data
import torch
from torch.utils.data import Dataset
from utils.claim import Claim
import re
token_pattern = re.compile(r"(?u)\b\w\w+\b")
tokenizer = lambda doc: token_pattern.findall(doc)


class ClaimDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self._data = load_data(data_path)
        self._vocab = {word: index for index, word in enumerate(load_vocab())}

    def preprocess(self, doc):
        return torch.tensor([self._vocab.get(token, 0) for token in tokenizer(doc.lower())])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        claim = Claim.from_json_object(self._data[idx])
        claim_text_preprocessed = self.preprocess(claim.claim)
        # claim_articles_preprocessed = list(map(self.preprocess, claim.related_articles_text))
        claim_articles_preprocessed = self.preprocess(' '.join(claim.related_articles_text))
        return (claim_text_preprocessed, claim_articles_preprocessed), claim.label


if __name__ == '__main__':
    dataset = ClaimDataset('data/train/train.json')
    print(dataset[0])