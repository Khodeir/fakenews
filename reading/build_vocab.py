from nltk.tokenize import word_tokenize
from collections import defaultdict
from utils.claim import Claim, load_data
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def data_iterator():
	data = load_data()
	for obj in data:
		claim = Claim.from_json_object(obj)
		yield claim.claim

		for article in claim.related_articles_text:
			yield article

def count_vectorizer_data_vocab():
	vectorizer = CountVectorizer()
	counts = vectorizer.fit_transform(data_iterator())

	with open('data/data_vocab_count_vectorizer.pkl', 'wb') as save_fp:
		pickle.dump(dict(vectorizer=vectorizer, counts=counts), save_fp)

# def simple_collect_data_vocab():
# 	data = data_iterator()
# 	word_dict = defaultdict(int)
# 	for document in data:

# 		for word in word_tokenize(document):
# 			word_dict[word.lower()] += 1

# 	with open('data/word_dict.pkl', 'wb') as save_fp:
# 		pickle.dump(word_dict, save_fp)

# def load_simple_data_vocab():
# 	with open('data/word_dict.pkl', 'rb') as save_fp:
# 		vocab = pickle.load(save_fp)
# 	return vocab

def load_count_vectorizer_data_vocab():
	with open('data/data_vocab_count_vectorizer.pkl', 'rb') as save_fp:
		result = pickle.load(save_fp)
	return result

def load_glove_model(glove_file='data/glove.6B.100d.txt'):
    f = open(glove_file,'r')
    model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    return model

def select_limited_vocab(limit=50000):
	# glove_embeddings = load_glove_model()
	data_vocab = load_count_vectorizer_data_vocab()
	count_vectorizer = data_vocab['vectorizer']
	counts = data_vocab['counts']
	tfidf = TfidfTransformer().fit(counts)
	response = tfidf.transform(counts)
	tfidf_sorting = np.argsort(np.asarray(response.sum(0))).flatten()[::-1]
	feature_array = np.array(count_vectorizer.get_feature_names())
	with open('data/vocab.txt', 'wb') as vocab_file:
		vocab_file.write('<UNK>\n'.encode('utf8'))
		for word in feature_array[tfidf_sorting[:limit]]:
			vocab_file.write('{}\n'.format(word).encode('utf8'))

def load_vocab():
	with open('data/vocab.txt', 'rb') as vocab_file:
		data = vocab_file.read().decode('utf8')
	return data.split('\n')
	# for word in data_vocab:
def bootstrap_embeddings_with_glove(embedding_size=100):
	vocab = load_vocab()
	embeddings = np.random.random((len(vocab), 100))
	model = load_glove_model()
	overlap = 0
	for i, word in enumerate(vocab):
		if word in model:
			overlap += 1
			embeddings[i] = model[word]
	print('overlap with glove: {}'.format(overlap))
	with open('data/embeddings.pkl', 'wb') as embeddings_file:
		pickle.dump(embeddings, embeddings_file)

def load_embeddings():
	with open('data/embeddings.pkl', 'rb') as embeddings_file:
		embeddings = pickle.load(embeddings_file)
	return embeddings

if __name__ == '__main__':
	# count_vectorizer_data_vocab()
	# select_limited_vocab()
	bootstrap_embeddings_with_glove()
