import torch
from torch.utils.data import DataLoader
from reading.data import ClaimDataset
from reading.model import AttentiveClassifier
from reading.build_vocab import load_embeddings
from tensorboardX import SummaryWriter
from functools import partial
from torch.nn.utils.rnn import pad_sequence

writer = SummaryWriter('models/reader_debug')
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
    batch_size=32,
    collate_fn=collate_fn
)

model = AttentiveClassifier(
    num_classes=3,
    vocab_size=embeddings.shape[0],
    embedding_dim=embeddings.shape[1],
    initial_embeddings=torch.as_tensor(embeddings, dtype=torch.float),
    hidden_dim=100,
    lstm_layers=1,
    lstm_bidirectional=True
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0
)

criterion = torch.nn.CrossEntropyLoss(weights=[0.692920, 0.795714, 3.026621])

num_epochs = 10
num_iters = 0
for epoch in range(num_epochs):
    for (
        (claim_text, document_text),
        label,
    ) in train_data:
        _, class_logits = model(
            claim_text,
            document_text
        )
        loss = criterion(class_logits, label)
        print(loss)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        num_iters += 1

        if writer is not None:
            writer.add_scalar(
                "loss", loss, num_iters
            )
