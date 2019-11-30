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

    def __init__(self, input_ids, attention_mask, token_type_ids, article_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.article_id = article_id

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
        return self._create_examples(data_dir, "train_split_v2_dupd.json")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev_split_v2.json")
    
    def truncate_to_example(self, claim):
        start = claim.find('See\xa0Example(s)')
        if start == -1:
            return claim
        else:
            return claim[:start].strip()
    
    def get_test_examples(self, data_dir):
        """Creates test examples for inference."""
        examples = []
        path = os.path.join(data_dir, 'metadata.json')
        df = pd.read_json(path)
        for (_, row) in df.iterrows():
            claim = row['claim']
            claimant = row['claimant']
            date = row['date']
            claim_id = row['id']
            label = 0 # placeholder
            article_ids = row['related_articles']

            # date_str = date.strftime("%B %d, %Y")
            claim = self.truncate_to_example(claim)
            text_a = claimant + ' : ' + claim
            text_a = text_a[:500].strip() 
            articles = []
            for article_id in article_ids:
                articles_path = os.path.join(
                    data_dir,
                    f'articles/{article_id}.txt')
                with open(articles_path) as f:
                    txt = f.read().strip()
                article_example = ArticleInputExample(
                    article_id=article_id, text_a=text_a, text_b=txt, label=label)
                articles.append(article_example)

            examples.append(
                InputExample(guid=claim_id, text_a=text_a, articles=articles, label=label))
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
            articles = []
            for article_id in article_ids:
                articles_path = os.path.join(
                    data_dir,
                    f'train_articles/{article_id}.txt')
                with open(articles_path) as f:
                    txt = f.read().strip()
                article_example = ArticleInputExample(
                    article_id=article_id, text_a=text_a, text_b=txt, label=label)
                articles.append(article_example)

            examples.append(
                InputExample(guid=claim_id, text_a=text_a, articles=articles, label=label))
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

        if ex_index % 10000 == 0:
            logger.info("Writing claim example %d of %d" % (ex_index, len(examples)))
        if ex_index < 5:
            also_print = True
        else:
            also_print = False
        article_features = _article_examples_to_features(
                            example.articles, tokenizer,
                            max_length,
                            pad_on_left,
                            pad_token,
                            pad_token_segment_id,
                            mask_padding_with_zero,
                            also_print=also_print)
        
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("claim_id: %s, label: %s (id = %d)" % (example.guid, example.label, label_id))

        features.append(
            InputFeatures(
                article_features=article_features,
                label_id=label_id,
                guid=example.guid))

    return features

def _article_examples_to_features(examples, tokenizer,
                                  max_length=512,
                                  pad_on_left=False,
                                  pad_token=0,
                                  pad_token_segment_id=0,
                                  mask_padding_with_zero=True,
                                  also_print=False):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
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
        if also_print:
            logger.info("*** Article Example ***")
            logger.info("article_id: %s" % (example.article_id))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        features.append(
            ArticleInputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                article_id=example.article_id
            )
        )

    return features
'''
def OLD_article_examples_to_features(examples, label_list, max_seq_length,
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
'''

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
        feature = self.features[idx]
        afs = feature.article_features
        # Cap at 8 articles per claim?
        all_input_ids = torch.tensor([f.input_ids for f in afs[:8]], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in afs[:8]], dtype=torch.long)
        all_segment_ids = torch.tensor([f.token_type_ids for f in afs[:8]], dtype=torch.long)
        label_id = torch.tensor(feature.label_id, dtype=torch.long)
        guid = feature.guid
        return (all_input_ids, all_input_mask, all_segment_ids, label_id, guid)
