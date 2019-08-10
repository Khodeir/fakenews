import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TaskSpecificAttention(nn.Module):
    def __init__(self, input_size, projection_size):
        super(TaskSpecificAttention, self).__init__()
        self.input_size = input_size
        self.projection_size = projection_size
        self.context_vector = torch.randn((1, 1, projection_size), requires_grad=True)
        self.input_projection = nn.Tanh(nn.Linear(input_size, projection_size))
        self.softmax = nn.Softmax()

    def forward(self, input_seq):
        '''inputs should be [seq_length, batch_size, input_size]'''
        vector_attention = self.input_projection(input_seq) # should be [seq_length, batch_size, output_size]
        attention_weights = self.softmax((vector_attention * self.context_vector).sum(2, keepdim=True), dim=0) # should be [seq_length, batch_size, 1]
        return attention_weights

class BiLinearAttention(nn.Module):
    def __init__(self, input_size):
        super(BiLinearAttention, self).__init__()
        self.input_size = input_size

        self.context_matrix = nn.Parameter(torch.randn(input_size, input_size))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, question, document):
        '''inputs should be [seq_length, batch_size, input_size]'''
        document_context = torch.matmul(document, self.context_matrix)
        attention_matrix = question.transpose(0, 1).bmm(document_context.permute(1, 2, 0))
        thing = attention_matrix.sum(1, keepdim=True).permute(2,0,1)
        attention_weights = self.softmax(thing)
        return attention_weights


class AttentiveReader(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, lstm_layers=1, lstm_bidirectional=True, initial_embeddings=None):
        super(AttentiveReader, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2 if lstm_bidirectional else 1

        if initial_embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(initial_embeddings, freeze=False)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.question_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers, bidirectional=lstm_bidirectional)
        self.document_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers, bidirectional=lstm_bidirectional)
        self.attention = BiLinearAttention(input_size=self.output_dim)

    def forward(self, question, document):
        question_embedding = self.word_embeddings(question)
        document_embedding = self.word_embeddings(document)

        # TODO: Maybe i only need the last states of the lstm
        question_encoding, _ = self.question_lstm(question_embedding)
        document_encoding, _ = self.document_lstm(document_embedding)
        # print('encoding sizes', question_encoding.shape, document_encoding.shape)

        attention = self.attention(question_encoding, document_encoding)
        output = (attention * document_encoding).sum(0)

        return output, attention

class AttentiveClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        num_classes = kwargs.pop('num_classes')
        self.attentive_reader = AttentiveReader(*args, **kwargs)
        self.classifier = nn.Linear(self.attentive_reader.output_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, question, document):
        logits, _ = self.attentive_reader(question, document)
        logits = self.classifier(logits)
        classes = self.softmax(logits)

        return classes, logits

if __name__ == '__main__':
    m = AttentiveReader(embedding_dim=20, hidden_dim=10, vocab_size=100, lstm_layers=1, lstm_bidirectional=True)
    question_batch = torch.randint(0, 100, (4, 10))
    document_batch = torch.randint(0, 100, (4, 200))
    out, _ = m(question_batch, document_batch)
    assert out.shape[0] == 4 and out.shape[1] == 20 and len(out.shape) == 2, "Got shape {}".format(out.shape)


    m = AttentiveClassifier(num_classes=3, embedding_dim=20, hidden_dim=10, vocab_size=100, lstm_layers=1, lstm_bidirectional=True)
    question_batch = torch.randint(0, 100, (4, 10))
    document_batch = torch.randint(0, 100, (4, 200))
    out, logits = m(question_batch, document_batch)
    assert out.shape[0] == 4 and out.shape[1] == 3 and len(out.shape) == 2, "Got shape {}".format(out.shape)


