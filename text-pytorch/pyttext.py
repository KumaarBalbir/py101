# For notes: refer info.txt

from transformers import BertTokenizer, BertForSequenceClassification
from torchmetrics.text import ROUGEScore
from torchmetrics.text import BLEUScore
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
from torchmetrics import Precision, Recall
from torchmetrics import ConfusionMatrix
from torchmetrics import F1Score
from torchmetrics import Accuracy
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import torch
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from torchtext.data.utils import get_tokenizer
import re

# ************************************ PREPROCESSING ************************************

# Tokenization using torchtext
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(
    "I am going to participate in a hackathon, I love writing programs that solve problems.")
print(tokens)  # ["I","am","going","to", ..]

# stop word removal
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokens = ["I", "am", "going", ...]
filtered_tokens = [token for token in tokens if token.lower()
                   not in stop_words]
print(filtered_tokens)  # ["going","participate",..]

# stemming
stemmer = PorterStemmer()
filtered_tokens = ["going", "participate", ...]
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print(stemmed_tokens)  # ["go","participate",..]

# remove infrequent words that don't add value
stemmed_tokens = ["go", "participate", ...]
freq_dist = FreqDist(stemmed_tokens)
threshold = 2
common_tokens = [
    token for token in stemmed_tokens if freq_dist[token] > threshold]


# ************************************ ENCODING ************************************

# One-hot encoding
vocab = ['cat', 'dog', 'rabbit']
vocab_size = len(vocab)
# eye provides a matrix with ones on the main diagonal
one_hot_vectors = torch.eye(vocab_size)
one_hot_dict = {word: one_hot_vectors[i] for i, word in enumerate(vocab)}
# {'cat': tensor([1., 0., 0.]), 'dog': tensor([0., 1., 0.]), 'rabbit': tensor([0., 0., 1.])}
print(one_hot_dict)

# Bag of words
# Using CountVectorizer

vectorizer = CountVectorizer()
corpus = ['This is the first document.', 'This document is the second document.',
          'And this is the third one.', 'Is this the first document?']
X = vectorizer.fit_transform(corpus)
print(X.toarray())
# ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
print(vectorizer.get_feature_names_out())

# TF-IDF
# using TfidfVectorizer

vectorizer = TfidfVectorizer()
corpus = ['This is the first document.', 'This document is the second document.',
          'And this is the third one.', 'Is this the first document?']
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())

# Embedding: We will see it later


# ************************************ DATASET & DATALOADER ************************************
# Dataset as a container for processed and encoded text
# Dataloader: batching, shuffling and multiprocessing

# creat a class


class TextDataset(Dataset):
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]


dataset = TextDataset(encoded_text)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Using helper functions


def preprocess_sentences(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokens = tokenizer(sentence)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        freq_dist = FreqDist(tokens)
        threshold = 2
        tokens = [token for token in tokens if freq_dist[token] > threshold]
        processed_sentences.append(' '.join(tokens))

    return processed_sentences


def encode_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    encoded_sentences = X.toarray()
    return encoded_sentences, vectorizer


def extract_sentences(data):
    sentences = re.findall(r'[A-Z][^.!?]*[.!?]', data)  # extract sentences
    return sentences


# *************************** Text processing pipeline: raw data > preprocessing > encoding > dataset & dataloader*********
def text_processing_pipeline(text):
    tokens = preprocess_sentences(text)
    encoded_sentences, vectorizer = encode_sentences(tokens)
    dataset = TextDataset(encoded_sentences)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    return dataloader, vectorizer


# apply the text processing pipeline
text_data = "This is the first text data. This is the second text data. And this is the third text data. Is this the first text data?"
sentences = extract_sentences(text_data)
dataloader, vectorizer = [text_processing_pipeline(text) for text in sentences]

print(next(iter(dataloader))[0, :10])

# text classification: binary, multiclass, multilabel (each text can be assigned to multiple classes)

# ******************************  Word Embedding in PyTorch ***************************
# torch.nn.Embedding : creates word vectors from indexes
# input: a tensor of word indexes
# output: a tensor of word vectors

words = ["The", "cat", "sat", "on", "the", "mat"]
word_to_idx = {word: i for i, word in enumerate(words)}
inputs = torch.LongTensor([word_to_idx[w] for w in words])
# TODO: how to decide embedding_dim?
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=10)
output = embedding(inputs)
print(output)

# Using embeddings in the pipeline


def preprocess_sentences(text):
    # tokenization
    # stemming
    ...
    # word to index mapping
    pass


text = "Your sample text"
dataloader, vectorizer = text_processing_pipeline(text)
# TODO: How to choose num_embeddings and embedding_dim?
embedding = nn.Embedding(num_embeddings=10, embedding_dim=50)

for batch in dataloader:
    inputs = batch
    output = embedding(inputs)
    print(output)


# ************************* CNN : text classification in PyTorch *******************

class SentimentAnalysisCNN (nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()  # initialize the parent class

        # create an embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=32,
                              kernel_size=3, stride=1, padding=1)  # create a convolutional layer
        # create a fully connected layer
        self.fc = nn.Linear(in_features=32, out_features=2)

    def forward(self, text):
        # permute the dim of the input tensor, 0:batch, 1:sequence, 2:embedding_dim
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = F.relu(self.conv(embedded))
        pooled = F.max_pool1d(conved, conved.shape[2])
        pooled = pooled.squeeze(2)  # remove the dim of size 1
        return self.fc(pooled)


# prepare data for sentiment analysis
vocab = ["i", "love", "this", "book", "it", "is",
         "a", "great", "book", "do", "not", "like", "it"]
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)
embed_dim = 10
book_samples = [
    # .split() converts a string into a list of words
    ("The story was captivating and kept me hooked until the end.".split(), 1),
    ("The characters were engaging and well-written.".split(), 1),
    ("I found the storyline interesting and engaging.".split(), 1),
    ("I had a hard time understanding the plot.".split(), 0),
    ("The plot was confusing and the characters were confusing.".split(), 0),
    ("I found the characters confusing and the storyline confusing.".split(), 0),
]
model = SentimentAnalysisCNN(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


data = book_samples
# train the model
for epoch in range(100):
    for sentence, label in data:
        model.zero_grad()  # zero the gradient TODO: why do we zero the gradient?
        sentence = torch.LongTensor([word_to_idx.get(w, 0) for w in sentence]).unsqueeze(
            0)  # TODO: why unsqueeze(0)?
        label = torch.LongTensor([int(label)])
        loss = criterion(model(sentence), label)
        loss.backward()
        optimizer.step()


# use the model to predict
for sample in book_samples:
    input_tensor = torch.tensor([word_to_idx[w]
                                for w in sample], dtype=torch.long).unsqueeze(0)
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs.data, 1)
    sentiment = "positive" if predicted.item() == 1 else "negative"
    print(f"Book Review: {' '.join(sample)}")
    print(f"Sentiment: {sentiment}")

# LSTM Arch: Input gate forget gate, and output gate


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # TODO: why passing LSTMModel to super(..)? Ans:
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return output

# GRU Arch


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.fc(hidden.squeeze(0))
        return output


# Evaluation metrics
# Accuracy
actual = torch.tensor([0, 1, 1, 0, 1, 0])
predicted = torch.tensor([0, 1, 0, 0, 1, 0])
accuracy = Accuracy(task="binary", num_classes=2)
acc = accuracy(actual, predicted)
print(acc)

# Precision and recall
precision = Precision(task="binary", num_classes=2)
recall = Recall(task="binary", num_classes=2)
prec = precision(actual, predicted)
rec = recall(actual, predicted)
print(prec)
print(rec)

# F1 score
f1 = F1Score(task="binary", num_classes=2)
f1 = f1(actual, predicted)
print(f1)

# Confusion matrix
cm = ConfusionMatrix(task="binary", num_classes=2)
cm = cm(actual, predicted)
print(cm)


# **************** RNN: text classification in PyTorch ********************
data = "Hello how are you?"
chars = list(set(data))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        # TODO: why batch_first=True?
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # we are initializing the hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        # TODO: why out[:,-1,:]? Ans: it's the last hidden state
        out = self.fc(out[:, -1, :])
        return out


model = RNNModel(1, 16, 1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# preparing input and target data
# creating indexes -> tensor conversion -> one-hot encoding -> target preparation

inputs = [char_to_ix[ch] for ch in data[:-1]]
targets = [char_to_ix[ch] for ch in data[1:]]

inputs = torch.tensor(inputs, dtype=torch.long).view(-1,
                                                     1)  # TODO: why view(-1,1)?
inputs = nn.functional.one_hot(inputs, num_classes=len(chars)).float()

targets = torch.tensor(targets, dtype=torch.long)

# training
for epoch in range(100):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch+1}/100, Loss: {loss.item()}")

# testing the model
model.eval()
test_input = char_to_ix['h']
test_input = nn.functional.one_hot(torch.tensor(
    test_input).view(-1, 1), num_classes=len(chars)).float()

predicted_output = model(test_input)
predicted_char_ix = torch.argmax(predicted_output, 1).item()
print(f"Predicted character: {model(test_input).item()}")

# ********************** GAN: in pytorch ************************


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: nn.Sequential() used for?
        self.model = nn.Sequential(nn.Linear(seq_length, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_length, seq_length), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


# initialize network and loss fn
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()

optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# training the discriminator
num_epochs = 100
for epoch in range(num_epochs):
    for real_data in data:
        real_data = real_data.unsqueeze(0)  # TODO: why unsqueeze(0)?
        noise = torch.rand((1, seq_length))
        disc_real = discriminator(real_data)
        fake_data = generator(noise)
        disc_fake = discriminator(fake_data.detach())
        loss_disc = criterion(disc_real, torch.ones_like(disc_real))+criterion(
            disc_fake, torch.zeros_like(disc_fake))  # TODO: meaning of this loss?
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

# training the generator
        disc_fake = discriminator(fake_data)
        loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

    if epoch+1 % 10 == 0:
        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss D: {loss_disc.item()}, Loss G: {loss_gen.item()}")

# print real and gen data
print(f"Real data: {real_data}")
print(data[:5])

print(f"Generated data: ")
for _ in range(5):
    noise = torch.rand((1, seq_length))
    generated_data = generator(noise)
    print(torch.round(generated_data).detach())

# GPT2 text generation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
seed_text = "Once upon a time"
# TODO: why return_tensors='pt'?
input_ids = tokenizer.encode(seed_text, return_tensors='pt')
output = model.generate(input_ids, max_length=100, temperature=0.7, no_repeat_ngram_size=2,
                        pad_token_id=tokenizer.eos_token_id)  # TODO: meaning of all these parameters?
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# T5 language translation
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
input_prompt = "translate English to French: 'Hello, how are you?'"
# TODO: why return_tensors='pt'?
input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=100, temperature=0.7, no_repeat_ngram_size=2,
                        pad_token_id=tokenizer.eos_token_id)  # TODO: meaning of all these parameters?
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# **********************  Evaluating metrics for text generation ************************

# calculating BLEU score with pytorch

generated_text = ['the cat is on the mat']
real_text = [['there is a cat on the mat',
              'the cat is on the mat', 'a cat is on the mat']]

bleu = BLEUScore()
bleu_metric = bleu(generated_text, real_text)
print("BLEU Score: ", bleu_metric.item())

# calculating ROUGE score with pytorch

generated_text = ['the cat is on the mat']
real_text = [['there is a cat on the mat',
              'the cat is on the mat', 'a cat is on the mat']]

rouge = ROUGEScore()
rouge_metric = rouge(generated_text, real_text)
print("ROUGE Score: ", rouge_metric.item())

# ********************* Transfer learning ************************
# implementing BERT
texts = [
    'I love this!',
    'I hate this!',
    'Amazing experience!',
    'Horrible experience!'
]
labels = [1, 0, 1, 0]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)

inputs = tokenizer(texts, return_tensors='pt', padding=True,
                   truncation=True, max_length=32)
inputs["labels"] = torch.tensor(labels)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(3):
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
text = "I had an awesome experience"
input_eval = tokenizer(text, return_tensors='pt',
                       padding=True, truncation=True, max_length=128)
outputs_eval = model(**input_eval)  # TODO: why **input_eval ?
predictions = torch.nn.functional.softmax(outputs_eval.logits, dim=-1)
predicted_label = 'positive' if torch.argmax(predictions) > 0 else 'negative'
print(f"Text: {text}\nSentiment: {predicted_label}")

# *********************** Transformers in PyTorch ************************
# prepare our data: train-test split
sentences = [
    "I love this product", "this is a great product",
    "this is a terrible product"
    "could be better"
]
labels = [1, 1, 0, 0]
train_sentences = sentences[:3]
train_labels = labels[:3]
test_sentences = sentences[3:]
test_labels = labels[3:]

# Building the transformer model


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


model = TransformerEncoder(embed_size=512, heads=8, num_layers=3, dropout=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# training the model
for epoch in range(100):
    for sentence, label in zip(train_sentences, train_labels):
        tokens = sentence.split()
        data = torch.stack([token_embeddings[token]
                           for token in tokens], dim=1)
        output = model(data)
        loss = criterion(output, torch.tensor([label]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# predicting the model


def predict(sentence):
    model.eval()
    with torch.no_grad():
        tokens = sentence.split()
        data = torch.stack([token_embeddings.get(
            token, torch.rand((1, 512))) for token in tokens], dim=1)
        output = model(data)
        predicted = torch.argmax(output, dim=1)
        return "positive" if predicted.item() == 1 else "negative"


# predicting on new text
sample_sentence = "this product could be better"
print(predict(sample_sentence))

# attention mechanism - setting vocal and data
data = ["the cat sat on the mat", ...]
vocab = set(' '.join(data).split())
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

pairs = [sentence.split() for sentence in data]
input_data = [[word_to_idx[word] for word in sentence[:-1]]
              for sentence in pairs]
target_data = [word_to_idx[sentence[-1]] for sentence in pairs]
inputs = [torch.tensor(seq, dtype=torch.long) for seq in input_data]
targets = [torch.tensor(seq, dtype=torch.long) for seq in target_data]

# Model definition
embeddng_dim = 10
hidden_dim = 16


class RNNWithAttentionModel(nn.Module):
    def __init__(self):
        super(RNNWithAttentionModel, self).__init__()
        self.embeddings = nn.Embedding(len(vocab), embeddng_dim)
        self.rnn = nn.RNN(embeddng_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, len(vocab))

    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.rnn(x)
        attn_weights = torch.nn.functional.softmax(
            self.attention(out).squeeze(2), dim=1)
        context = torch.sum(attn_weights.unsqueeze(2) * out, dim=1)
        out = self.fc(context)
        return out

    def pad_sequences(batch):
        max_len = max([len(seq) for seq in batch])
        return torch.stack([torch.cat([seq, torch.zeros(max_len - len(seq)).long()], dim=0) for seq in batch])


# training prep
criterion = nn.CrossEntropyLoss()
attention_model = RNNWithAttentionModel()
optimizer = torch.optim.Adam(attention_model.parameters(), lr=0.001)
for epoch in range(100):
    attention_model.train()
    optimizer.zero_grad()
    padded_inputs = pad_sequences(inputs)
    outputs = attention_model(padded_inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Model evaluation
for input_seq, target in zip(input_data, target_data):
    input_test = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
    attention_model.eval()
    output = attention_model(input_test)
    predicted = idx_to_word[torch.argmax(output).item()]
    if predicted.item() == target:
        print("correct")
    else:
        print("incorrect")
