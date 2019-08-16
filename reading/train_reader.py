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
PATH = "models/reader_debug"
writer = SummaryWriter(PATH)
embeddings = load_embeddings()

dataset = ClaimDataset(
    data_path='data/train.json'
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

def collate_fn(batch):
    if len(batch) == 1:
        x, y = batch[0]
        x = tuple(map(partial(torch.unsqueeze, dim=0), x))
        y = y.unsqueeze(0)
        return x, y
    else:
        x, y = zip(*batch)
        y = torch.as_tensor(y)
        claims, docs = zip(*x)
        claims = pad_sequence(list(map(torch.as_tensor, claims)))
        docs = pad_sequence(list(map(torch.as_tensor, docs)))
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

optimizer = torch.optim.Adam(
    model.parameters(), lr=5e-4, weight_decay=0
)

criterion = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([0.692920, 0.795714, 3.026621]).to(device)
)

num_epochs = 20
num_iters = 0
torch.save(model.state_dict(), os.path.join(PATH, 'model_{}'.format(0)))
for epoch in range(num_epochs):
    print('Epoch {}'.format(num_iters))
    for (
        (claim_text, document_text),
        label,
    ) in train_data:
        claim_text = claim_text.to(device)
        document_text = document_text.to(device)
        label = label.to(device)
        _, class_logits = model(
            claim_text,
            document_text
        )
        loss = criterion(class_logits, label)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss.cpu().detach().numpy())
        num_iters += 1
        if writer is not None:
            writer.add_scalar(
                "loss", loss, num_iters
            )
        if num_iters % 10 == 0:
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), os.path.join(PATH, 'model_{}'.format(epoch)))