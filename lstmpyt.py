import torch
import config
import spacy
import time
from model import BERTBaseUncased
from torchtext.legacy import data
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
from torchtext.vocab import Vectors


train_data = pd.read_csv('train_folds.csv')
train_df, valid_df = train_test_split(train_data, test_size=0.2)
nlp = spacy.load('en_core_web_sm')
TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
LABEL = data.LabelField(dtype = torch.float, batch_first=True)


class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.target if not is_test else None
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, False, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


fields = [('text',TEXT), ('label',LABEL)]

train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df)

vectors = Vectors(name='./crawl-300d-2M.vec', cache='./')
MAX_VOCAB_SIZE = 100000
TEXT.build_vocab(train_ds, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = vectors,
                 unk_init = torch.Tensor.zero_)
LABEL.build_vocab(train_ds)




class LSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
               bidirectional, dropout, pad_idx):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
    self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                       bidirectional=bidirectional,
                       dropout=dropout,
                       batch_first=True)
    self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, 1)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, text, text_lengths):
    embedded = self.embedding(text)
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
    packed_output, (hidden,cell) = self.rnn(packed_embedded)

    hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    output = self.fc1(hidden)
    output = self.dropout(self.fc2(output))
    return output








