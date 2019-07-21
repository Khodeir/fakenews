DATA_DIR = 'data/train'
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