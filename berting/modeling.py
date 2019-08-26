import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_transformers import BertPreTrainedModel, BertModel

class BertForMultiSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForMultiSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_chunks, labels):
        chunk_outputs = []
        for chunk in input_chunks:
            input_ids = chunk[0]
            position_ids = chunk[1]
            token_type_ids = chunk[2]
            attention_mask = chunk[3]
            head_mask = chunk[4]

            outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask)
            pooled_output = outputs[1]
            chunk_outputs.append(pooled_output)
        
        mean = torch.mean(torch.stack(chunk_outputs))
        mean = self.dropout(mean)
        logits = self.classifier(mean)

        outputs = (logits,) 
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits