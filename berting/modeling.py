import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_transformers import BertPreTrainedModel, BertModel

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
        # hack it to be (num_articles, max_seq_length)
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
            loss_fct = CrossEntropyLoss(
                weight=torch.tensor([0.692920, 0.795714, 3.026621]).to(device)
            )
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits