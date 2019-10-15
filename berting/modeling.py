import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (BertPreTrainedModel, BertModel,
                          RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
                          RobertaModel)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertForMultiSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForMultiSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        # input_ids was meant to be (batch_size, max_seq_length)
        # hack it to be (1, num_articles, max_seq_length)
        input_ids = input_ids[0]
        token_type_ids = token_type_ids[0] if token_type_ids is not None else None
        attention_mask = attention_mask[0] if attention_mask is not None else None
        # labels dim is fine as is
        position_ids = position_ids[0] if position_ids is not None else None
        head_mask = head_mask[0] if head_mask is not None else None
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_outputs = outputs[1]
        # should be size (num_articles, hidden)
        mean = pooled_outputs.mean(dim=-2)
        mean = self.dropout(mean)
        logits = self.classifier(mean).view(1, self.num_labels)

        outputs = (logits,) 
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits

class RobertaCustomClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaCustomClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size * 8, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
        #self.lstm = nn.LSTM(config.hidden_size, config.hidden_size)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # should be size (num_articles, hidden)
        # x = x.mean(dim=-2) # avg cls representation across articles
        x = x.flatten()
        x2 = torch.zeros(self.config.hidden_size * 8).to(device)
        x2[:x.size(0)] = x
        #lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        #x2 = lstm_out[-1]
        x2 = self.dropout(x2)
        x2 = self.dense(x2)
        x2 = torch.tanh(x2)
        x2 = self.dropout(x2)
        x2 = self.out_proj(x2)
        return x2


class RobertaForMultiSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultiSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaCustomClassificationHead(config)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        # input_ids was meant to be (batch_size, max_seq_length)
        # hack it to be (1, num_articles, max_seq_length)
        input_ids = input_ids[0]
        token_type_ids = token_type_ids[0] if token_type_ids is not None else None
        attention_mask = attention_mask[0] if attention_mask is not None else None
        # labels dim is fine as is
        position_ids = position_ids[0] if position_ids is not None else None
        head_mask = head_mask[0] if head_mask is not None else None
        
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output).view(1, self.num_labels)

        outputs = (logits,) 
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits
