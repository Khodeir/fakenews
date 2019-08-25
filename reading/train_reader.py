import os
import torch
from torch.utils.data import DataLoader
from reading.data import ClaimDataset
from reading.model import AttentiveClassifier
from reading.build_vocab import load_embeddings
from tensorboardX import SummaryWriter
from functools import partial
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.environ.get('MODEL_DIR', None)
RESUME_FROM = os.environ.get('RESUME_FROM', None)
writer = SummaryWriter(MODEL_DIR)
embeddings = load_embeddings()

dataset = ClaimDataset(
    data_path='data/train.json'
)

def collate_fn(batch):
    if len(batch) == 1:
        x, y = batch[0]
        x = tuple(map(partial(torch.unsqueeze, dim=0), x))
        y = y.unsqueeze(0)
        return x, y
    else:
        _, x, y = zip(*batch)
        y = torch.as_tensor(y)
        claims, docs = zip(*x)
        claims = pad_sequence(
            [torch.tensor(claim, dtype=torch.long) for claim in claims])
        n_docs = max([len(doc) for doc in docs])
        batch_docs = [[] for i in range(n_docs)]
        for doc in docs:
            for i in range(n_docs):
                d = doc[i] if len(doc) > i else []
                batch_docs[i].append(d)
        
        docs = [pad_sequence([torch.tensor(d, dtype=torch.long) for d in doc])
                for doc in batch_docs]
        return (claims, docs), y
        # x = sorted(x, key=len)

    raise ValueError

train_data = DataLoader(
    dataset,
    shuffle=True,
    batch_size=64,
    collate_fn=collate_fn,
    num_workers=2
)
model = AttentiveClassifier(
    num_classes=3,
    vocab_size=embeddings.shape[0],
    embedding_dim=embeddings.shape[1],
    initial_embeddings=torch.tensor(embeddings, dtype=torch.float32),
    hidden_dim=200,
    lstm_layers=2,
    lstm_bidirectional=True
).to(device)
if RESUME_FROM:
    print('Resuming from {}'.format(RESUME_FROM))
    model.load_state_dict(torch.load(RESUME_FROM))
    name = os.path.split(RESUME_FROM)[-1].replace(".pt", "")
    epoch = int(name.split('_')[-2]) + 1
    num_iters = int(name.split('_')[-1]) + 1
else:
    epoch = 0
    num_iters = 0
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4, weight_decay=0
)

criterion = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([0.692920, 0.795714, 3.026621]).to(device)
)

num_epochs = 20
for epoch in range(epoch, epoch + num_epochs):
    print('Epoch {}'.format(num_iters))
    for (
        (claim_text, document_text),
        label,
    ) in train_data:
        claim_text = claim_text.to(device)
        document_text = [d.to(device) for d in document_text]
        label = label.to(device)
        _, class_logits = model(
            claim_text,
            *document_text
        )
        loss = criterion(class_logits, label)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(name, param.grad, num_iters) 
        print(loss.cpu().detach().numpy())
        num_iters += 1
        if writer is not None:
            writer.add_scalar(
                "loss", loss, num_iters
            )
        if num_iters % 10 == 0:
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'model_{}_{}.pt'.format(epoch, num_iters)))
