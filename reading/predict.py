import os
from reading.data import ClaimDataset
import torch
import pandas as pd
from reading.model import load_model

def load_data(path):
    dataset = ClaimDataset(
        data_path=path
    )
    dataset.return_ids = True
    eval_data = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        # batch_size=1,
        shuffle=True,
        # collate_fn=collate_fn
    )
    return eval_data

def model_predict(model, eval_data, limit):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total = 0
    ids = []
    preds = []
    for (
        i,
        (claim_text, document_text),
        _,
    ) in eval_data:
        claim_text = torch.tensor(claim_text).to(device)
        document_text = torch.tensor(document_text).unsqueeze(0).to(device)
        probs, _ = model(
            claim_text.transpose(1, 0),
            document_text.transpose(1, 0)
        )
        total += 1
        pred = probs.argmax(dim=1)
        preds.extend(pred.cpu().numpy())
        ids.extend(i.numpy())
        if total > limit:
            break

    return dict(ids=ids, preds=preds)


if __name__ == '__main__':
    
    JSON_PATH = os.environ.get('JSON_PATH', None)
    RESUME_FROM = os.environ.get('RESUME_FROM', None)
    LIMIT = int(os.environ.get('EVAL_LIMIT', 1000000))
    DF_PATH = os.environ.get('DF_PATH', None)
    dataset = load_data(JSON_PATH)
    model = load_model(RESUME_FROM)
    model.eval()
    df = model_predict(model, dataset, limit=LIMIT)
    df = pd.DataFrame(df)
    df.to_csv(DF_PATH, header=False, index=False, columns=['ids', 'preds'])
