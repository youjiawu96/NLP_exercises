import pickle, argparse, os, sys, bcolz
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.modules import loss
import torch.optim as optim
import re

def preprocess(training_file, glove_path = '.'):
    # First read in the file and preprocess:
    with open(training_file, 'rb') as f:
        contents = f.readlines()

    words = []
    tags = []
    vocab = {}
    uniq_word = {}
    uniq_tag = {}
    tag_idx = 1
    uniq_tag['<PAD>'] = 0
    uniq_word['<PAD>'] = 0
    for k, sentence in enumerate(contents):
        # omit the last '\n' and the first 'b''
        sentence = str(sentence)
        sentence = sentence[2:-3]
        # get rid of possible lines starting with space
        i = 0
        while sentence[i] == ' ':
            i += 1
        sentence = sentence[i:]
        parsed_sen = re.split(r'\s+', sentence)
        sub_tags = []
        for i in range(len(parsed_sen)//2):
            # vocab counts the word apprearance in the training set, to determine which words should be labelled as "UNKA"
            if parsed_sen[2*i] not in vocab:
                vocab[parsed_sen[2*i]] = 1
                # word_idx += 1
                # sub_words.append(vocab[parsed_sen[2*i]])
            else:
                vocab[parsed_sen[2*i]] += 1
            # uniq_tag assigns every tag a number, tags saves each tag sequence
            # print(k, i, parsed_sen[2*i+1])
            if parsed_sen[2*i+1] not in uniq_tag:
                uniq_tag[parsed_sen[2*i+1]] = tag_idx
                tag_idx += 1
                sub_tags.append(uniq_tag[parsed_sen[2*i+1]])

            else:
                sub_tags.append(uniq_tag[parsed_sen[2*i+1]])
        # append the tag sequence to the tags
        tags.append(sub_tags)
    uniq_word['UNKA'] = 1
    word_idx = 2
    for sentence in contents:
        sentence = str(sentence)
        sentence = sentence[2:-3]
        # get rid of possible lines starting with space
        i = 0
        while sentence[i] == ' ':
            i += 1
        sentence = sentence[i:]
        parsed_sen = re.split(r'\s+',sentence)
        sub_words = []
        for i in range(len(parsed_sen)//2):
            if vocab[parsed_sen[2*i]] <= 3:
                sub_words.append(uniq_word['UNKA'])
            elif parsed_sen[2*i] not in uniq_word:
                uniq_word[parsed_sen[2*i]] = word_idx
                word_idx += 1
                sub_words.append(uniq_word[parsed_sen[2*i]])
            else:
                sub_words.append(uniq_word[parsed_sen[2*i]])

        words.append(sub_words)
    

    # now create the embedding matrix using GloVe pre-trained model data
    # read the pre-stored GloVe files
    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    glove_words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in glove_words}
    emb_dim = 50
    matrix_len = len(uniq_word)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0

    for word in uniq_word:
        try: 
            weights_matrix[uniq_word[word]] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[uniq_word[word]] = np.random.normal(scale=0.6, size=(emb_dim, ))

    return words, tags, uniq_word, uniq_tag, weights_matrix

    
class RNNTagger(nn.Module):
    def __init__(self, training_file, embedding_dim = 50, hidden_dim = 100):
        super(RNNTagger, self).__init__()
        self.hidden_dim = hidden_dim
        _, _, self.vocab, self.tags, self.weights_matrix = preprocess(training_file)
        n_vocab_words = len(self.vocab)
        padding_idx = self.vocab['<PAD>']
        # embedding layer, use the pretrained weight matrix, make sure the weights are not trainable
        # let the embedding know that padding is '0'
        self.word_embedding = nn.Embedding(num_embeddings=n_vocab_words,embedding_dim=embedding_dim,padding_idx=padding_idx)
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.weights_matrix))
        self.word_embedding.weight.requires_grad = False
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, len(self.tags))
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        # input sentence here is already the embedded sentence, batch_size*len(sentence)*embedding_dim
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.logsoftmax(tag_space)
        return tag_scores

def train(training_file):
    assert os.path.isfile(training_file), 'Training file does not exist'

    # Your code starts here
    # check whether the GloVe embedding is already processed, if not, process it.
    glove_path = '.'
    if not os.path.isfile(f'{glove_path}/6B.50_words.pkl'):
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

        with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
            
        vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
        vectors.flush()
        pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
        pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

    training_X, training_Y, _, _, _ = preprocess(training_file)
    model = RNNTagger(training_file, hidden_dim=100)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.05)
    
    with torch.no_grad():
        inputs = torch.tensor(training_X[0], dtype=torch.long)
        tag_scores = model(inputs)
        print(tag_scores)
    
    for j in range(10):
        print("current training epoch #", j)
        for i in range(len(training_X)):
            # clear out the gradients from last step
            model.zero_grad()

            tag_scores = model(torch.tensor(training_X[i], dtype=torch.long))

            loss = loss_function(tag_scores, torch.tensor(training_Y[i], dtype=torch.long))
            if i%2000 == 0:
                print("current loss is: ", loss)
            loss.backward()
            optimizer.step()
    # Your code ends here

    return model

def test(model_file, data_file, label_file):
    assert os.path.isfile(model_file), 'Model file does not exist'
    assert os.path.isfile(data_file), 'Data file does not exist'
    assert os.path.isfile(label_file), 'Label file does not exist'

    # Your code starts here
    # check whether the GloVe embedding is already processed, if not, process it.
    glove_path = '.'
    if not os.path.isfile(f'{glove_path}/6B.50_words.pkl'):
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

        with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
            
        vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
        vectors.flush()
        pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
        pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))
    
    model = RNNTagger("./wsj1-18.training", hidden_dim=100)
    model.load_state_dict(torch.load(model_file))
    with open(data_file, 'rb') as f:
        contents = f.readlines()
    test_input = []
    for k, sentence in enumerate(contents):
        # omit the last '\n' and the first 'b''
        sentence = str(sentence)
        sentence = sentence[2:-3]
        # get rid of possible lines starting with space
        i = 0
        while sentence[i] == ' ':
            i += 1
        sentence = sentence[i:]
        parsed_sen = re.split(r'\s+', sentence)
        for w in parsed_sen:
            if w in model.vocab:
                test_input.append(model.vocab[w])
            else:
                # if the word is not present in the vocabulary, treat it as 'UNKA'
                test_input.append(1)
    # input tensor: as a 1 row tensor containing all the words
    test_input_tensor = torch.tensor(test_input, dtype=torch.long)
            
    prediction = model(test_input_tensor)
    prediction = torch.argmax(prediction,-1).cpu().numpy()

    with open(label_file, 'rb') as f:
        lab_contents = f.readlines()
    ground_truth = []
    for k, sentence in enumerate(lab_contents):
        # omit the last '\n' and the first 'b''
        sentence = str(sentence)
        sentence = sentence[2:-3]
        # get rid of possible lines starting with space
        i = 0
        while sentence[i] == ' ':
            i += 1
        sentence = sentence[i:]
        parsed_sen = re.split(r'\s+', sentence)
        for l in range(len(parsed_sen)//2):
            if parsed_sen[2*l+1] in model.tags:
                ground_truth.append(model.tags[parsed_sen[2*l+1]])
            else:
                # if seeing a tag not seen before, assign it to a random previous tag
                ground_truth.append(random.randint(0,len(model.tags)-1))

    # Your code ends here

    print(f'The accuracy of the model is {100*accuracy_score(prediction, ground_truth):6.2f}%')

def main(params):
    if params.train:
        model = train(params.training_file)
        torch.save(model.state_dict(), params.model_file)
    else:
        test(params.model_file, params.data_file, params.label_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    parser.add_argument("--train", action='store_const', const=True, default=False)
    parser.add_argument('--model_file', type=str, default='model.torch')
    parser.add_argument('--training_file', type=str, default='')
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--label_file', type=str, default='')

    main(parser.parse_args())
