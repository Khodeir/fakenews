import pandas as pd
import numpy as np
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)
import torch
from torch.utils.data import Dataset


class ArticleInputExample(object):

    def __init__(self, claim_id, text_a, text_b, label=None):
        self.claim_id = claim_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputExample(object):

    def __init__(self, guid, text_a, articles, label=None):
        self.guid = guid
        self.text_a = text_a
        self.articles = articles
        self.label = label


class ArticleInputFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, claim_id, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.claim_id = claim_id
        self.label_id = label_id

class InputFeatures(object):

    def __init__(self, article_features, label_id, guid):
        self.article_features = article_features
        self.label_id = label_id
        self.guid = guid


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class FakeNewsProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train_dupd.json")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev_split_v2.json")
    
    def truncate_to_example(self, claim):
        start = claim.find('See\xa0Example(s)')
        if start == -1:
            return claim
        else:
            return claim[:start].strip()
    
    def get_test_examples(self, data_dir, model_dir):
        """Creates test examples for inference."""
        examples = []
        path = os.path.join(data_dir, 'metadata.json')
        df = pd.read_json(path)

        sent_mdl = torch.load(os.path.join(model_dir, 'sentModel'))
        sent_mdl = sent_mdl.cuda()
        def cosine(u, v):
            return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        
        for (_, row) in df.iterrows():
            claim = row['claim']
            claimant = row['claimant']
            # date = row['date']
            claim_id = row['id']
            label = 0 # placeholder
            article_ids = row['related_articles']

            # date_str = date.strftime("%B %d, %Y")
            claim = self.truncate_to_example(claim)
            text_a = claimant + ' : ' + claim
            text_a = text_a[:500].strip()

            # condense begin
            u = sent_mdl.encode([claim])[0]
            sentences = []
            for aid in article_ids:
                articles_path = os.path.join(
                    data_dir,
                    f'articles/{aid}.txt')
                with open(articles_path) as f:
                    for line in f:
                        line = line.strip()
                        if len(line) > 5:
                            sentences.append(line[:5000])
            print(f'len of sents is {len(sentences)}')
            embeddings = sent_mdl.encode(sentences, bsize=32, tokenize=True, verbose=False)
            #print(embeddings.shape)

            sims = [cosine(u, v) for v in embeddings]
            
            if len(sims) > 5:
                ind = np.argpartition(sims, -5)[-5:]
            else:
                ind = list(range(len(sims)))

            condensed_txt = '\n'.join([sentences[idx] for idx in sorted(ind)])
            torch.cuda.empty_cache()
            
            # final step
            article_example = ArticleInputExample(
                claim_id=claim_id, text_a=text_a, text_b=condensed_txt, label=label)
            examples.append(article_example)

        return examples

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    def _create_examples(self, data_dir, file_name):
        """Creates examples for the training and dev sets."""
        examples = []
        path = os.path.join(data_dir, file_name)
        df = pd.read_json(path)
        for (_, row) in df.iterrows():
            claim = row['claim']
            claimant = row['claimant']
            date = row['date']
            claim_id = row['id']
            label = row['label']
            article_ids = row['related_articles']

            # date_str = date.strftime("%B %d, %Y")
            claim = self.truncate_to_example(claim)
            text_a = claimant + ' : ' + claim
            text_a = text_a[:500].strip()
            
            articles_path = os.path.join(
                data_dir,
                f'condensed_train_articles_global/{claim_id}_top5_txt.txt')
            with open(articles_path) as f:
                txt = f.read().strip()
            article_example = ArticleInputExample(
                claim_id=claim_id, text_a=text_a, text_b=txt, label=label)
            examples.append(article_example)
        return examples

def convert_examples_to_features(examples, tokenizer,
                                 max_length=512,
                                 task='fn',
                                 label_list=None,
                                 output_mode=None,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        logger.info("Writing claim example %d of %d" % (ex_index, len(examples)))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncate_first_sequence=False  # We're truncating the SECOND sequence in priority
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        try: 
            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
        except Exception as e:
            print(f'Length ERROR! article id is {example.article_id}')
            print(f'text a is {example.text_a}')
            print(f'text b is {example.text_b}')
            raise e

        if ex_index < 5:
            logger.info("*** Article Example ***")
            logger.info("claim_id: %s" % (example.claim_id))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        
        logger.info("claim_id: %s, label: %s (id = %d)" % (example.claim_id, example.label, label_id))
        
        features.append(
            ArticleInputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id,
                claim_id=example.claim_id
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    print(f'Preds are {preds}')
    print(f'Labels are {labels}')
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, labels=[0, 1, 2], average=None)
    print(f'f1 is {f1}')
    #macro_avg_f1 = f1_score(y_true=labels, y_pred=preds, labels=[0, 1, 2], average='macro')
    precision = precision_score(y_true=labels, y_pred=preds, labels=[0, 1, 2], average=None)
    recall = recall_score(y_true=labels, y_pred=preds, labels=[0, 1, 2], average=None)
    P = (precision[0] + precision[1] + precision[2])/3
    R = (recall[0] + recall[1] + recall[2])/3
    macro_avg_f1 = 2 * P * R / (P + R)
    return {
        "acc": acc,
        "f1_0": f1[0],
        "f1_1": f1[1],
        "f1_2": f1[2],
        "precision_0": precision[0],
        "precision_1": precision[1],
        "precision_2": precision[2],
        "recall_0": recall[0],
        "recall_1": recall[1],
        "recall_2": recall[2],
        "macro_avg_f1": macro_avg_f1

    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "fn":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

processors = {
    "fn": FakeNewsProcessor,
}

output_modes = {
    "fn": "classification",
}

class FakeNewsDataset(Dataset):
    def __init__(self, input_features):
        self.features = input_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        all_input_ids = torch.tensor(f.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(f.attention_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(f.token_type_ids, dtype=torch.long)
        label_id = torch.tensor(f.label_id, dtype=torch.long)
        guid = f.claim_id
        return (all_input_ids, all_input_mask, all_segment_ids, label_id, guid)
