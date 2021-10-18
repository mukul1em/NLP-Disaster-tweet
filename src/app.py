import streamlit as st
import torch
import config
import spacy
import time
from model import BERTBaseUncased
from torchtext.legacy import data
import torch.nn as nn
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





DEVICE = torch.device('cpu')



def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )
    
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]



nlp = spacy.load('en_core_web_sm')
def bilstm(model, sentence):
  tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
  indexed = [TEXT.vocab.stoi[t] for t in tokenized]
  length = [len(indexed)]
  tensor = torch.LongTensor(indexed).to(device)
  tensor = tensor.unsqueeze(1).T
  length_tensor = torch.LongTensor(length)
  prediction = model(tensor, length_tensor).squeeze(1)

  rounded_preds = torch.round(torch.sigmoid(prediction))
  predict_class = rounded_preds.tolist()[0]
  return predict_class


st.title('Classify whether given tweet is a disaster or not!!!')

sentence = st.text_input('enter your sentence here: ')
button = st.button('predict')

option = st.sidebar.selectbox('select  model',
                        ('BERT','BiLSTM'))


    


if option == 'BERT':
    st.text('High Positive value indicates disaster tweet ')
    st.text('High Negative value indicates not a disaster tweet')

    if sentence:
        if button:
            MODEL_PATH = './model-2.bin'

            MODEL = BERTBaseUncased()
            MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            MODEL.to(DEVICE)
            start_time = time.time()
            with st.spinner(text="PREDICTING.."):
                positive_prediction = sentence_prediction(sentence)
                negative_prediction = 1 - positive_prediction
                response = {}
                response["response"] = {
                    "positive": str(positive_prediction),
                    "negative": str(negative_prediction),
                    "sentence": str(sentence),
                    "time_taken": str(time.time() - start_time),
                }      
                st.write(response)
                st.balloons()
                st.success('DONE')


elif option == 'BiLSTM':
    if sentence:
        if button:

            train_data = pd.read_csv('../input/train_folds.csv')
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

            INPUT_DIM = len(TEXT.vocab)
            EMBEDDING_DIM = 300
            HIDDEN_DIM = 256
            OUTPUT_DIM = 1
            N_LAYERS = 2
            BIDIRECTIONAL = True
            DROPOUT = 0.2
            PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


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



            model= LSTM(INPUT_DIM, 
                        EMBEDDING_DIM, 
                        HIDDEN_DIM, 
                        OUTPUT_DIM, 
                        N_LAYERS, 
                        BIDIRECTIONAL, 
                        DROPOUT, 
                        PAD_IDX)


            device = torch.device('cpu')


            pretrained_embeddings = TEXT.vocab.vectors
            model.embedding.weight.data.copy_(pretrained_embeddings)

            #  to initiaise padded to zeros
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
            with st.spinner(text="PREDICTING.."):
                model = torch.load('./lstmmodel.bin', map_location=torch.device('cpu'))


                output = bilstm(model, sentence)
                if output == 0.0:
                    st.subheader("response: Not a disaster tweet")
                else:
                    st.subheader("response: Disaster tweet")
                st.balloons()
                st.success('DONE')           
