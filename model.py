import transformers
import config
import torch.nn as nn

#define model
class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            config.BERT_PATH, return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1) #bertbaseuncased has 768 out layers

    def forward(self, ids, mask, token_type_ids):
        _,o2 = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        ) 
        bo = self.bert_drop(o2)
        output = self.out(bo) 
        return output








         
