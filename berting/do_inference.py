import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from pytorch_transformers import (BertConfig, BertTokenizer)
from modeling import BertForMultiSequenceClassification

from data_providers import (convert_examples_to_features,
                            output_modes, processors, FakeNewsDataset)

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", default='/usr/local/dataset/metadata.json', type=str, required=False,
                    help="Path to json file with claim id, claim, and article ids.")
parser.add_argument("--model_dir", default='/usr/src', type=str, required=False,
                    help="Path to pretrained model.")
parser.add_argument("--article_dir", default='/usr/local/dataset/articles', type=str, required=False,
                    help="Path to dir for articles.")
parser.add_argument("--output_file_path", default='/usr/local/predictions.txt', type=str, required=False,
                    help="File path for output file.")
parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int, required=False,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--max_seq_length", default=512, type=int, required=False,
                    help="Max sequence length for inference.")
parser.add_argument("--model_type", default='bert', type=str, required=False,
                    help="Model type.")

args = parser.parse_args()

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMultiSequenceClassification, BertTokenizer),
}

# Prepare data
args.task_name = 'fn'
if args.task_name not in processors:
    raise ValueError("Task not found: %s" % (args.task_name))
processor = processors[args.task_name]()
output_mode = output_modes[args.task_name]
label_list = processor.get_labels()
num_labels = len(label_list)

config = config_class.from_pretrained(f'{args.model_dir}/multi_article_cased_2e-5', num_labels=num_labels, finetuning_task=args.task_name)
tokenizer = tokenizer_class.from_pretrained(f'{args.model_dir}/multi_article_cased_2e-5', do_lower_case=False)
model = model_class.from_pretrained(f'{args.model_dir}/multi_article_cased_2e-5', config=config)
model.eval()
model = model.to('cuda')

def load_and_cache_examples(args, tokenizer, processor, output_mode, label_list):
    examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
        pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
    dataset = FakeNewsDataset(features)
    return dataset

test_dataset = load_and_cache_examples(args, tokenizer, processor, output_mode, label_list)

args.eval_batch_size = args.per_gpu_eval_batch_size * 1
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

# Test!
print("***** Running test *****")
print("  Num examples = %d", len(test_dataset))
print("  Batch size = %d", args.eval_batch_size)

results = []
for batch in tqdm(test_dataloader, desc="Testing"):
    model.eval()
    batch = tuple(t.to('cuda') for t in batch)

    with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids 
                 }
        outputs = model(**inputs)
        logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    claim_ids = batch[4].detach().cpu().numpy()
    assert len(logits) == len(claim_ids)
    for claim_id, logits_ex in zip(claim_ids, logits):
        if output_mode == "classification":
            pred = np.argmax(logits_ex)
        elif output_mode == "regression":
            #pred = np.squeeze(logits_ex)
            raise ValueError('Output mode shouldnt be regression!')
        row = {
            'claim_id':  int(claim_id),
            'pred': int(pred)
        }
        results.append(row)

df = pd.DataFrame(results)
df = df[['claim_id', 'pred']]
df.to_csv(args.output_file_path, index=False, header=False)
