
import pandas as pd
import numpy as np

from torchtext.data import Field, TabularDataset, BucketIterator
import torchtext.data.functional as f
from torchtext.data.utils import get_tokenizer
import spacy
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm.notebook import tqdm
tqdm.pandas()

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

######### Loading from TSV files and cleaning data #########

# STEPS:
# 0. Remove line separators and punctuation, lemmatize text. Remove unused data.
# Stopwords to be remained as potentially important part. Save processed data.
# 1. Specify how preprocessing should be done through torchtext.data.Field
# 2. Use TebularDataset to load the data from files
# 3. Construct an iterator to do batching & padding through torchtext.data.BucketIterator

### Data Loading
train = pd.read_csv('../input/uci-drug-review-dataset/drugLibTrain_raw.tsv', sep='\t')
train.dropna(how='any', inplace = True)
test = pd.read_csv('../input/uci-drug-review-dataset/drugLibTest_raw.tsv', sep='\t')
text_columns = ['benefitsReview','sideEffectsReview', 'commentsReview']

for col in text_columns:
    # removing Windows line separators
    train[col] = train[col].progress_apply(lambda x: x.replace('\r\r\n',''))
    test[col] = test[col].progress_apply(lambda x: x.replace('\r\r\n',''))

### Review cleaning 
nlp = spacy.load('en_core_web_sm')
def clear_review(sentence):
    """Function to lemmatize words and remove punctuation 
    Args:
    sentence: sentence to process, str
    Return:
    clean sentence, str
    """
    words = nlp(sentence)
    s = [''.join(word.lemma_) for word in words if word.is_punct == False and
        word.is_digit==False and word.pos_ != 'PRON']
    return ' '.join(map(str,s))

for col in text_columns:
    train[col] = train[col].progress_apply(lambda x: clear_review(x).lower().strip())
    test[col] = test[col].progress_apply(lambda x: clear_review(x).lower().strip())

### Separating text features and target
df_train = train.loc[:,['benefitsReview','sideEffectsReview', 'commentsReview','rating']]
df_test = test.loc[:,['benefitsReview','sideEffectsReview', 'commentsReview','rating']]
# Full review collection to make vocab out of it
X = df_train.append(df_test).drop('rating',axis=1)

### Train/test split and saving to new files 
# df_train, df_val = train_test_split(df_train, test_size = 0.05,
#                                     random_state = 42)
# df_train.to_csv('df_train.csv')
# df_val.to_csv('df_val.csv')
# df_test.to_csv('df_test.csv')

### Data Preprocessing. Dataset and Iterators creation.

def tokenize(text):
    """Spacy tokenization function"""
    return [tok.text for tok in nlp.tokenizer(text)]


review = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
rating = Field(sequential=False, use_vocab=False)

fields = {"benefitsReview": ("bnf", review), "sideEffectsReview": ("sef", review),
          "commentsReview": ("cmt", review), "rating": ("rtg", rating)}

train_data, val_data, test_data = TabularDataset.splits(
    path="mydata", train="df_train.csv",validation = 'df_val.csv', test="df_train.csv", format="csv",
    fields=fields)


#Making full vocabulary out of every review
pr_benefits = X['benefitsReview'].apply(lambda x: review.preprocess(x))
pr_effects = X['sideEffectsReview'].apply(lambda x: review.preprocess(x))
pr_comments = X['commentsReview'].apply(lambda x: review.preprocess(x))
preprocessed_text = pr_benefits.append([pr_effects,pr_comments])

print('Creating vocabulary')
review.build_vocab(
    preprocessed_text, 
    vectors='glove.6B.300d'
)

vocab = review.vocab
print(f'Vocabulary created. Its shape is: {vocab.vectors.shape}')

#Iterators Creation
train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data,val_data, test_data), batch_size=64, device=device,
    sort = False
)

######### Defining model and helper functions #########

# STEPS:

class RNN_GRU(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embed_size)
        self.do = nn.Dropout(p=0.2)
        self.rnn1 = nn.GRU(embed_size, hidden_size, num_layers)
        self.rnn2 = nn.GRU(embed_size, hidden_size, num_layers)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, 1)


    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
  
    def forward(self, x):
        # Initializing hidden units and cells state

        embedded = self.embedding(x)
        do = self.do(embedded)
        outputs, _ = self.rnn1(do)
        # outputs, _ = self.rnn2(do)
        # outputs = self.relu(outputs)
        prediction = self.fc_out(outputs[-1, :, :])

        return prediction
  

    def train(self, train_iterator, num_epochs, criterion, optimizer, val_iterator=None):
        log = []
        prev_loss = 9999
        for epoch in range(num_epochs):
            print(f'{epoch} epoch started')
            train_losses = []
            for batch_idx, batch in enumerate(train_iterator):
                # Get data to cuda if possible
                # Here we're trying to take in account all text features
                data_bnf = batch.bnf.to(device=device)
                data_sef = batch.sef.to(device=device)
                data_cmt = batch.cmt.to(device=device)
        
                targets = batch.rtg.to(device=device)

                # forward
                pr_bnf = self(data_bnf)
                pr_sef = self(data_sef)
                pr_cmt = self(data_cmt)
                prediction = (pr_bnf+pr_sef+pr_cmt) / 3
                loss = criterion(prediction.squeeze(1).float(), targets.float())

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent
                optimizer.step()
                train_losses.append(loss.item())
            print('train loss on epoch {} : {:.3f}'.format(epoch, np.mean(train_losses)))
            if val_iterator is not None:
                test_losses = []
                for t_batch in val_iterator:
                    with torch.no_grad():
                        # forward
                        data_bnf = t_batch.bnf.to(device=device)
                        data_sef = t_batch.sef.to(device=device)
                        data_cmt = t_batch.cmt.to(device=device)
                        targets = t_batch.rtg.to(device=device)
                        pr_bnf = self(data_bnf)
                        pr_sef = self(data_sef)
                        pr_cmt = self(data_cmt)
                        prediction = (pr_bnf+pr_sef+pr_cmt) / 3
                        loss = criterion(prediction.squeeze(1).float(), targets.float())
                    #calculating test loss
                    test_losses.append(loss.item())
                new_loss = np.mean(test_losses)    
                print('test loss on epoch {}: {:.3f}'.format(epoch, new_loss))
                scheduler.step(new_loss)
                # making checkpoint if there is an improvement
                if new_loss < prev_loss:
                    print(f'making checkpoint on loss value: {new_loss}')
                    prev_loss = new_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': new_loss,
                    }, 'model.ckpt')
            # logging losses to make plot        
            log.append([np.mean(train_losses), np.mean(test_losses)])
            pd.DataFrame(log, columns=['train','val']).to_csv('train_log.csv')
    
    def evaluate(self, test_iterator):
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        with torch.no_grad():
            for batch in test_iterator:
                # loading predictors
                data_bnf = batch.bnf.to(device=device)
                data_sef = batch.sef.to(device=device)
                data_cmt = batch.cmt.to(device=device)
                #targets
                targets = batch.rtg.to(device=device)
                # predicting
                pr_bnf = self(data_bnf)
                pr_sef = self(data_sef)
                pr_cmt = self(data_cmt)
                prediction = (pr_bnf+pr_sef+pr_cmt) / 3
                predlist = torch.cat([predlist,prediction.view(-1).cpu()])
                lbllist = torch.cat([lbllist,targets.view(-1).cpu()])
        #calculating confusion matrix, class accuracy and mean accuracy        
        conf_mat=confusion_matrix(lbllist.numpy(), predlist.type(torch.uint8).numpy())
        class_accuracy=conf_mat.diagonal()/conf_mat.sum(1)
        return class_accuracy[1:], np.mean(class_accuracy[1:])

######### Deep Learning Setup #########

# Hyperparameters
input_size = len(review.vocab)
hidden_units = 512
num_layers = 2
embedding_size = 300
learning_rate = 0.001
num_epochs = 100

loss_function = nn.MSELoss()
model = RNN_GRU(input_size, embedding_size, hidden_units, num_layers).to(device)
model.init_weights()

# loading pretrained embeddings to our model
pretrained_embeddings = review.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                              factor=0.1, patience=0, verbose=True)


######### Training and evaluation #########
model.train(train_iterator, num_epochs, loss_function, optimizer, test_iterator)
class_acc, mean_class_acc = model.evaluate(val_iterator)
print('According to evaluation results, mean accuracy of model is: {mean_class_acc}')