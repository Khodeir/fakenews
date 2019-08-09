import os
import json

DATA_DIR = 'data/train'

def load_data(path=os.path.join(DATA_DIR, 'train.json')):
    with open(path, 'r') as json_fp:
        data = json.load(json_fp)
    return data

class Claim:
    _related_articles_text = None
    @classmethod
    def from_json_object(cls, json_object):
        claim = cls()
        for key in json_object:
            setattr(claim, key, json_object[key])
        return claim
    @property
    def related_articles_text(self):
        if self._related_articles_text:
            return self._related_articles_text

        self._related_articles_text = []
        for i in self.related_articles:
            with open('{}/train_articles/{}.txt'.format(DATA_DIR,i), 'rb') as related_article:
                self._related_articles_text.append(related_article.read().decode('utf8'))
        return self._related_articles_text