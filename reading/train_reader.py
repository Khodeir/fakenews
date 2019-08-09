import torch
from torch.utils.data import DataLoader
from reading.data import ClaimDataset
from reading.model import AttentiveClassifier
from reading.build_vocab import load_embeddings
from tensorboardX import SummaryWriter

writer = SummaryWriter('models/reader_debug')
embeddings = load_embeddings()

# TODO: somehow truncate or limit size of related docs
# Or maybe pass them as a list and iteratively compute classifications
dataset = ClaimDataset(
    data_path='data/train/train.json'
)

# TODO: add padding to support bigger batches
train_data = DataLoader(
    dataset, batch_size=1
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

# TODO: add weights to offset class imbalance
criterion = torch.nn.CrossEntropyLoss()

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
