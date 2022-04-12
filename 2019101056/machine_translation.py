# https://www.guru99.com/seq2seq-model.html
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nltk
import numpy as np
import pandas as pd
import string
import spacy
import os
import re
import random
import sys
import joblib

model = torch.load(sys.argv[1])
model.eval()
en_index2token = joblib.load('en_index2token.pkl')
fr_index2token = joblib.load('fr_index2token.pkl')
en_token2index = joblib.load('en_token2index.pkl')
fr_token2index = joblib.load('fr_token2index.pkl')
spacy_en = spacy.load('en_core_web_sm')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def french_sentence(input_sentence):
    input_tensor = torch.tensor(input_sentence, dtype=torch.long, device=device).view(-1, 1)
    target = np.zeros(len(input_sentence))
    output_tensor = torch.tensor(target, dtype=torch.long, device=device).view(-1, 1)
    # setting the teacher forcing ratio to be 0.0, so we get only the words predicted by the model     
    output = model(input_tensor, output_tensor,0.0)
    num_iter = output.size(0)
    decoded_words = []
    for ot in range(output.size(0)):
        topv, topi = output[ot].topk(1)
        # print(topi)

        if topi[0].item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(fr_index2token[topi[0].item()])

    return decoded_words


input_sentence = input('Input English Sentence: ')
input_sentence = input_sentence.lower()
lst = [tok.text for tok in spacy_en.tokenizer(input_sentence)]

sent = []
for token in lst:
    if en_token2index.get(token) == None:
        sent.append(en_token2index['unk'])
    else:
        sent.append(en_token2index[token])

print('Translated French Sentence: ',french_sentence(sent))
