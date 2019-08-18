import os
from reading.model import AttentiveClassifier
from reading.data import ClaimDataset
import torch
import numpy as np
from tensorboardX import SummaryWriter
from utils.metrics import acc_and_f1
import pandas as pd

def load_eval_data(path='data/train.json'):
    dataset = ClaimDataset(
        data_path=path
    )
    dataset.return_ids = True
    eval_data = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        shuffle=True
    )
    return eval_data

def load_model(path):
    model = AttentiveClassifier(
        num_classes=3,
        vocab_size=50002, # doesnt matter
        embedding_dim=100, # doesnt matter
        hidden_dim=200,
        lstm_layers=2,
        lstm_bidirectional=True
    )
    print('Resuming from {}'.format(path))
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def model_eval(model, eval_data, limit=np.inf):
    correct = 0
    total = 0
    ids = []
    labels = []
    preds = []
    for (
        i,
        (claim_text, document_text),
        label,
    ) in eval_data:
        claim_text = torch.tensor(claim_text)
        document_text = torch.tensor(document_text).unsqueeze(0)
        probs, _ = model(
            claim_text.transpose(1, 0),
            document_text.transpose(1, 0)
        )
        total += 1
        pred = probs.argmax(dim=1)
        labels.extend(label.numpy())
        preds.extend(pred.numpy())
        ids.extend(i.numpy())
        if pred == label:
            correct += 1

        if total % 10 == 0:
            print('Accuracy {}: {}'.format(total, correct/total))
        if total == limit:
            break
    metrics = acc_and_f1(np.array(preds), np.array(labels))

    return metrics, dict(ids=ids, labels=labels, preds=preds)

if __name__ == '__main__':
    DATA_PATH = os.environ.get('DATA_PATH', None)
    RESUME_FROM = os.environ.get('RESUME_FROM', None)
    LIMIT =  int(os.environ.get('EVAL_LIMIT', 1000000))
    DF_PATH = os.environ.get('DF_PATH', None)
    ITER = os.environ.get('ITER', False)
    dataset = load_eval_data(DATA_PATH)
    model = load_model(RESUME_FROM)
    metrics, df = model_eval(model, dataset, limit=LIMIT)
    df = pd.DataFrame(df)
    if DF_PATH:
        df.to_csv(DF_PATH)
    if ITER:
        writer = SummaryWriter(os.path.dirname(RESUME_FROM))
        for key in metrics:
            writer.add_scalar(key, metrics[key], int(ITER))
        writer.flush()

        
        