from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spacy
import random
import sys
import joblib

SOS_token = 0
EOS_token = 1
max_length = 25

# ENCODER DECODER SEQ2SEQ
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()

        #set the encoder input dimesion , embbed dimesion, hidden dimesion, and number of layers 
        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #initialize the embedding layer with input and embbed dimention
        self.embedding = nn.Embedding(input_dim, self.embbed_dim)
        #intialize the GRU to take the input dimetion of embbed, and output dimention of hidden and
        #set the number of gru layers
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
              
    def forward(self, src):

        embedded = self.embedding(src).view(1,1,-1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        #set the encoder output dimension, embed dimension, hidden dimension, and number of layers 
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # initialize every layer with the appropriate dimension. For the decoder layer, it will consist of an embedding, GRU, a Linear layer and a Log softmax activation function.
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
      
    def forward(self, input, hidden):
        # reshape the input to (1, batch_size)
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)       
        prediction = self.softmax(self.out(output[0]))

        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=max_length):
        super().__init__()
      
        #initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        
        input_length = source.size(0) #get the input length (number of words in sentence)
        batch_size = target.shape[1] 
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim

        #initialize a variable to hold the predicted outputs
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        #encode every word in a sentence
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[i])

        #use the encoder’s hidden layer as the decoder hidden
        decoder_hidden = encoder_hidden.to(device)

        #add a token before the first predicted word
        decoder_input = torch.tensor([SOS_token], device=device)  # SOS

        #topk is used to get the top K value over a list
        #predict the output word from the current target word. If we enable the teaching force,  then the #next decoder input is the next word, else, use the decoder output highest value. 

        for t in range(target_length):   
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teacher_force else topi)
            if(teacher_force == False and input.item() == EOS_token):
                break

        return outputs



en_index2token = joblib.load('en_index2token_q2.1.pkl')
fr_index2token = joblib.load('fr_index2token_q2.1.pkl')
en_token2index = joblib.load('en_token2index_q2.1.pkl')
fr_token2index = joblib.load('fr_token2index_q2.1.pkl')

# !python3 -m spacy download en_core_web_sm
spacy_en = spacy.load('en_core_web_sm')


input_size = len(en_index2token)
output_size = len(fr_index2token)
embed_size = 256
hidden_size = 512
num_layers = 1

#create encoder-decoder model
encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
decoder = Decoder(output_size, hidden_size, embed_size, num_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq2seq = Seq2Seq(encoder, decoder, device).to(device)
model = torch.load(sys.argv[1], map_location=device)
seq2seq.load_state_dict(model)
seq2seq.eval()



def french_sentence(input_sentence):
    input_tensor = torch.tensor(input_sentence, dtype=torch.long, device=device).view(-1, 1)
    target = np.zeros(len(input_sentence))
    output_tensor = torch.tensor(target, dtype=torch.long, device=device).view(-1, 1)
    # setting the teacher forcing ratio to be 0.0, so we get only the words predicted by the model     
    output = seq2seq(input_tensor, output_tensor,0.0)
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

print('Translated French Sentence: '," ".join(french_sentence(sent)))

