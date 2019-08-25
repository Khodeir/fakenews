from reading.build_vocab import load_vocab
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.claim import Claim, load_data
import re
token_pattern = re.compile(r"(?u)\b\w\w+\b")
tokenizer = lambda doc: token_pattern.findall(doc)


class ClaimDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self._data = load_data(data_path)
        self._vocab = {word: index for index, word in enumerate(load_vocab())}
        self._max_num_articles = 3
        self._max_words_per_article  = 300

    def preprocess(self, doc):
        return torch.tensor([self._vocab.get(token, 0) for token in tokenizer(doc.lower())])

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        claim = Claim.from_json_object(self._data[idx])
        claim_text_preprocessed = self.preprocess(claim.claim)
        claim_articles = claim.related_articles_text[:]
        if self._max_num_articles:
            claim_articles = np.random.choice(claim_articles, min(
                len(claim_articles), self._max_num_articles), replace=False)
        claim_articles_preprocessed = [
            self.preprocess(text) for text in claim_articles]
        if self._max_words_per_article:
            for idx in range(len(claim_articles_preprocessed)):
                article = claim_articles_preprocessed[idx]
                if len(article) > self._max_words_per_article:
                    start = np.random.randint(low=0, high=len(article) - self._max_words_per_article, size=1)[0]
                    claim_articles_preprocessed[idx] = article[start:start + self._max_words_per_article]
        
        return claim.id, (claim_text_preprocessed, claim_articles_preprocessed), claim.label


if __name__ == '__main__':
    dataset = ClaimDataset('data/train/train.json')
    print(dataset[0])
