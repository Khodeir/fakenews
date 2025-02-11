from data_providers import (convert_examples_to_features,
                            output_modes, processors, InputExample)
import spacy
print('spaCy Version: %s' % spacy.__version__)
spacy_nlp = spacy.load('en_core_web_md')

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)


parser = argparse.ArgumentParser()
parser.add_argument("--data_file", default=None, type=str, required=True,
                    help="Path to json file with claim id, claim, and article ids.")
args = parser.parse_args()

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}

config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
config = config_class.from_pretrained('/home/ubuntu/results/bert_base_uncased_sts-b', num_labels=len([None]), finetuning_task='sts-b')
tokenizer = tokenizer_class.from_pretrained('/home/ubuntu/results/bert_base_uncased_sts-b', do_lower_case=True)
model = model_class.from_pretrained('/home/ubuntu/results/bert_base_uncased_sts-b', from_tf=False, config=config)
model.eval()
model = model.to('cuda')

df = pd.read_json(args.data_file)

def get_article_txt(did, also_print=False):
    with open(f'/home/ubuntu/fakenews/data/train/train_articles/{did}.txt') as f:
        txt = f.read()
    if also_print:
        print(txt)
    return txt

for idx, row in df.iterrows():
    claim = row[0]
    claim_id = row[3]
    articles = row[5]
    
    examples = []
    for article_id in articles:
        text = get_article_txt(article_id, also_print=False)
        doc = spacy_nlp(text)
        for i, token in enumerate(doc.sents):
            if len(token.text) > 25:
                examples.append(InputExample(guid=article_id, text_a=claim, text_b=token.text, label=0))
    
    if len(examples) == 0:
        
        with open(f'/home/ubuntu/fakenews/data/train/condensed_v2/{claim_id}_top20_txt.txt', 'w+') as f:
            f.write('placeholder')
        with open(f'/home/ubuntu/fakenews/data/train/condensed_v2/{claim_id}_top20_scores.txt', 'w+') as f:
            f.write('placeholder')
        
        with open(f'/home/ubuntu/fakenews/data/train/condensed_v2b/{claim_id}_top20_txt.txt', 'w+') as f:
            f.write('placeholder')
        with open(f'/home/ubuntu/fakenews/data/train/condensed_v2b/{claim_id}_top20_scores.txt', 'w+') as f:
            f.write('placeholder')
        
        with open(f'/home/ubuntu/fakenews/data/train/condensed_v3/{claim_id}_top6_txt.txt', 'w+') as f:
            f.write('placeholder')
        with open(f'/home/ubuntu/fakenews/data/train/condensed_v3/{claim_id}_top6_scores.txt', 'w+') as f:
            f.write('placeholder')
    
    
    features = convert_examples_to_features(examples, [None], 128, tokenizer, 'regression',
            cls_token_at_end=False,            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=0,
            pad_on_left=False,                # pad on the left for xlnet
            pad_token_segment_id=0)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    preds = None
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to('cuda') for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      }
            outputs = model(**inputs)
            logits = outputs[0]
            #probs = F.softmax(logits, dim=1)
            if preds is None:
                #preds = probs.detach().cpu().numpy()
                preds = logits.detach().cpu().numpy()
            else:
                #preds = np.append(preds, probs.detach().cpu().numpy(), axis=0)
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    
    assert len(preds) == len(examples)
    flattened = preds.flatten()
    if len(flattened) > 21:
        indices = list(np.argpartition(flattened, -21)[-21:])
    else:
        indices = list(range(len(flattened)))
    top = np.argmax(flattened)
    if flattened[top] > 3.5:
        indices.remove(top)
    
    filtered = sorted(indices)
    
    condensed_txt = '\n'.join([examples[idx].text_b for idx in filtered])
    condensed_scores = '\n'.join([str(flattened[idx]) for idx in filtered])
        
    with open(f'/home/ubuntu/fakenews/data/train/condensed_v2/{claim_id}_top20_txt.txt', 'w+') as f:
        f.write(condensed_txt)
    with open(f'/home/ubuntu/fakenews/data/train/condensed_v2/{claim_id}_top20_scores.txt', 'w+') as f:
        f.write(condensed_scores)
    
    # second version
    if len(flattened) > 21:
        indices = list(np.argpartition(flattened, -21)[-21:])
    else:
        indices = list(range(len(flattened)))
    
    filtered = sorted([idx for idx in indices if flattened[idx] < 3.5])
    
    
    condensed_txt = '\n'.join([examples[idx].text_b for idx in filtered])
    condensed_scores = '\n'.join([str(flattened[idx]) for idx in filtered])
        
    with open(f'/home/ubuntu/fakenews/data/train/condensed_v2b/{claim_id}_top20_txt.txt', 'w+') as f:
        f.write(condensed_txt)
    with open(f'/home/ubuntu/fakenews/data/train/condensed_v2b/{claim_id}_top20_scores.txt', 'w+') as f:
        f.write(condensed_scores)
    
    # thrid version --- top 6 with context
    if len(flattened) > 7:
        indices = list(np.argpartition(flattened, -7)[-7:])
    else:
        indices = list(range(len(flattened)))
    # top = np.argmax(flattened)
    if flattened[top] > 3.5:
        indices.remove(top)
        
    filtered = []
    
    for idx in indices:
        if idx-1 >= 0:
            filtered.append(idx-1)
        filtered.append(idx)    
        if idx+1 <= len(examples)-1:
            filtered.append(idx+1)
    
    filtered = sorted(list(set(filtered)))
    
    condensed_txt = '\n'.join([examples[idx].text_b for idx in filtered])
    condensed_scores = '\n'.join([str(flattened[idx]) for idx in filtered])
        
    with open(f'/home/ubuntu/fakenews/data/train/condensed_v3/{claim_id}_top6_txt.txt', 'w+') as f:
        f.write(condensed_txt)
    with open(f'/home/ubuntu/fakenews/data/train/condensed_v3/{claim_id}_top6_scores.txt', 'w+') as f:
        f.write(condensed_scores)