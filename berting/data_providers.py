import pandas as pd
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

    def __init__(self, article_id, text_a, text_b, label=None):
        self.article_id = article_id
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

    def __init__(self, input_ids, input_mask, segment_ids, label_id, article_id, tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.article_id = article_id
        self.tokens = tokens

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
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train_split_12.json")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev_split.json")
    
    def get_test_examples(self, data_dir):
        """Creates test examples for inference."""
        examples = []
        path = os.path.join(data_dir, 'metadata.json')
        df = pd.read_json(path)
        for (_, row) in df.iterrows():
            claim = row[0]
            claimant = row[1]
            date = row[2]
            claim_id = row[3]
            label = 0 # placeholder
            article_ids = row[5]

            # date_str = date.strftime("%B %d, %Y")
            text_a = claimant + ' : ' + claim 
            articles = []
            for article_id in article_ids:
                articles_path = os.path.join(
                    data_dir,
                    f'articles/{article_id}.txt')
                with open(articles_path) as f:
                    txt = f.read()
                article_example = ArticleInputExample(
                    article_id=article_id, text_a=text_a, text_b=txt, label=label)
                articles.append(article_example)

            examples.append(
                InputExample(guid=claim_id, text_a=text_a, articles=articles, label=label))
        return examples


    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, data_dir, file_name):
        """Creates examples for the training and dev sets."""
        examples = []
        path = os.path.join(data_dir, file_name)
        df = pd.read_json(path)
        for (_, row) in df.iterrows():
            claim = row[0]
            claimant = row[1]
            date = row[2]
            claim_id = row[3]
            label = row[4]
            article_ids = row[5]

            # date_str = date.strftime("%B %d, %Y")
            text_a = claimant + ' : ' + claim 
            articles = []
            for article_id in article_ids:
                articles_path = os.path.join(
                    data_dir,
                    f'train_articles/{article_id}.txt')
                with open(articles_path) as f:
                    txt = f.read()
                article_example = ArticleInputExample(
                    article_id=article_id, text_a=text_a, text_b=txt, label=label)
                articles.append(article_example)

            examples.append(
                InputExample(guid=claim_id, text_a=text_a, articles=articles, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):

        if ex_index % 10000 == 0:
            logger.info("Writing claim example %d of %d" % (ex_index, len(examples)))
        if ex_index < 5:
            also_print = True
        else:
            also_print = False
        article_features = _article_examples_to_features(
            example.articles, label_list, max_seq_length,
            tokenizer, output_mode,
            cls_token_at_end, pad_on_left,
            cls_token, sep_token, pad_token,
            sequence_a_segment_id, sequence_b_segment_id,
            cls_token_segment_id, pad_token_segment_id,
            mask_padding_with_zero,
            also_print=also_print)
        
        label_id = label_map[example.label]

        features.append(
            InputFeatures(
                article_features=article_features,
                label_id=label_id,
                guid=example.guid))

    return features


def _article_examples_to_features(examples, label_list, max_seq_length,
                                  tokenizer, output_mode,
                                  cls_token_at_end=False, pad_on_left=False,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=0, pad_token_segment_id=0,
                                  mask_padding_with_zero=True,
                                  also_print=False):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
    
        if also_print:
            logger.info("*** Article Example ***")
            logger.info("article_id: %s" % (example.article_id))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
     
        features.append(
            ArticleInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                article_id=example.article_id,
                tokens=tokens
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
    macro_avg_f1 = f1_score(y_true=labels, y_pred=preds, labels=[0, 1, 2], average='macro')
    precision = precision_score(y_true=labels, y_pred=preds, labels=[0, 1, 2], average=None)
    recall = recall_score(y_true=labels, y_pred=preds, labels=[0, 1, 2], average=None)
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
        feature = self.features[idx]
        afs = feature.article_features
        # Cap at 8 articles per claim?
        all_input_ids = torch.tensor([f.input_ids for f in afs[:8]], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in afs[:8]], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in afs[:8]], dtype=torch.long)
        label_id = torch.tensor(feature.label_id, dtype=torch.long)
        guid = feature.guid
        return (all_input_ids, all_input_mask, all_segment_ids, label_id, guid)