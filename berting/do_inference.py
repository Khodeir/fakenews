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
from pytorch_transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer)


parser = argparse.ArgumentParser()
parser.add_argument("--data_file", default='/usr/local/dataset/metadata.json', type=str, required=True,
                    help="Path to json file with claim id, claim, and article ids.")
parser.add_argument("--model_dir", default='/usr/src/models', type=str, required=True,
                    help="Path to pretrained model.")
parser.add_argument("--article_dir", default='/usr/local/dataset/articles', type=str, required=True,
                    help="Path to dir for articles.")
parser.add_argument("--save_dir", default='/usr/src/condensed_articles', type=str, required=True,
                    help="Path to dir for saving.")
parser.add_argument("--output_file_path", default='/usr/local/predictions.txt', type=str, required=True,
                    help="File path for output file.")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, required=True,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--model_type", default='bert', type=str, required=True,
                    help="Model type.")

args = parser.parse_args()

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}

config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
config = config_class.from_pretrained(f'{args.model_dir}/bert_base_uncased_sts-b', num_labels=len([None]), finetuning_task='sts-b')
tokenizer = tokenizer_class.from_pretrained(f'{args.model_dir}/bert_base_uncased_sts-b', do_lower_case=True)
model = model_class.from_pretrained(f'{args.model_dir}/bert_base_uncased_sts-b', from_tf=False, config=config)
model.eval()
model = model.to('cuda')

df = pd.read_json(args.data_file)

def get_article_txt(did, also_print=False):
    with open(f'{args.article_dir}/{did}.txt') as f:
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
        with open(f'{args.save_dir}/{claim_id}_top20_txt.txt', 'w+') as f:
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
        
    with open(f'{args.save_dir}/{claim_id}_top20_txt.txt', 'w+') as f:
        f.write(condensed_txt)

# Prepare data
args.task_name = 'fakenews'
if args.task_name not in processors:
    raise ValueError("Task not found: %s" % (args.task_name))
processor = processors[args.task_name]()
args.output_mode = output_modes[args.task_name]
label_list = processor.get_labels()
num_labels = len(label_list)

config = config_class.from_pretrained(f'{args.model_dir}/condensed_v2_base_dup_2e-5', num_labels=num_labels, finetuning_task=args.task_name)
tokenizer = tokenizer_class.from_pretrained(f'{args.model_dir}/condensed_v2_base_dup_2e-5', do_lower_case=args.do_lower_case)
model = model_class.from_pretrained(f'{args.model_dir}/condensed_v2_base_dup_2e-5', config=config)
model.eval()
model = model.to('cuda')

def load_and_cache_examples(args, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    label_list = processor.get_labels()
    examples = processor.get_test_examples(args.data_file, args.save_dir)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
        pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_guids)
    return dataset

test_dataset = load_and_cache_examples(args, args.task_name, tokenizer)

args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
# Note that DistributedSampler samples randomly
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

# Test!
print("***** Running test *****")
print("  Num examples = %d", len(test_dataset))
print("  Batch size = %d", args.eval_batch_size)

results = []
for batch in tqdm(test_dataloader, desc="Testing"):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)

    with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] 
                 }
        outputs = model(**inputs)
        logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    claim_ids = batch[4].detach().cpu().numpy()
    assert len(logits) == len(claim_ids)
    for claim_id, logits_ex in zip(claim_ids, logits):
        if args.output_mode == "classification":
            pred = np.argmax(logits_ex)
        elif args.output_mode == "regression":
            pred = np.squeeze(logits_ex)
        row = {
            'claim_id':  int(claim_id),
            'pred': int(pred)
        }
        results.append(row)

df = pd.DataFrame(results)
df = df[['claim_id', 'pred']]
df.to_csv(args.output_file_path, index=False, header=False)