from build_vocab import load_embeddings, load_vocab

embeddings = load_embeddings()
vocab_size, embedding_dim = embeddings.shape
vocab = {word: index for index, word in enumerate(load_vocab())}
assert len(vocab) == vocab_size

token_pattern = re.compile(r"(?u)\b\w\w+\b")
tokenizer = lambda doc: token_pattern.findall(doc)

def preprocess(doc):
	return [vocab.get(token, 0) for token in tokenizer(doc.lower())]

def data_iterator():
	data = load_data()
	for obj in data:
		claim = Claim.from_json_object(obj)
		yield claim

preprocessed_arrays = 

