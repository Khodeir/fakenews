from reading.build_vocab import load_vocab, load_data
import torch
import numpy as np
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
        self._max_num_articles = 5
        self._max_words_total  = 500

    def preprocess(self, doc):
        return torch.tensor([self._vocab.get(token, 0) for token in tokenizer(doc.lower())])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        claim = Claim.from_json_object(self._data[idx])
        claim_text_preprocessed = self.preprocess(claim.claim)
        claim_articles = claim.related_articles_text[::]
        claim_articles = np.random.choice(claim_articles, min(len(claim_articles), self._max_num_articles))
        num_articles = len(claim_articles)
        words_left = self._max_words_total
        claim_articles_preprocessed = []
        for i, article in enumerate(claim_articles):
            claim_articles_preprocessed.extend(
                self.preprocess(article)[:(words_left // (num_articles - i))]
            )
            words_left = self._max_words_total - len(claim_articles_preprocessed)
        return (claim_text_preprocessed, claim_articles_preprocessed), claim.label


if __name__ == '__main__':
    dataset = ClaimDataset('data/train/train.json')
    print(dataset[0])