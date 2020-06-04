from torch.utils.data import Dataset
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from bs4 import BeautifulSoup
import re
import pytreebank
import json

word_to_ix = {}
with open("PQG/data/quora_data_prepro.json") as f:
    json_file = json.load(f)
    ix_to_word = json_file['ix_to_word']
    UNK_token = 0
    ix_to_word[UNK_token] = '<UNK>'
    EOS_token = len(ix_to_word)
    ix_to_word[EOS_token] = '<EOS>'
    PAD_token = len(ix_to_word)
    ix_to_word[PAD_token] = '<PAD>'
    SOS_token = len(ix_to_word)
    ix_to_word[SOS_token] = '<SOS>'
    vocab_size = len(ix_to_word)
    for key in ix_to_word:
        word_to_ix[ix_to_word[key]] = int(key)

sst = pytreebank.load_sst()

nltk.download('punkt')
nltk.download('wordnet')

class SSTDataset(Dataset):
    def __init__(self, split="train", seq_len=28, phrase=False):
        self.dataX, self.dataY = self.process_raw(sst[split], seq_len, phrase)

    def process_raw(self, forest, seq_len, phrase):
        idx = 0
        sents = []
        for tree in forest:
            if phrase:
                sents += tree.to_labeled_lines()
            else:
                sents.append(tree.to_labeled_lines()[0])
        sents = list(set(sents))
        dataX = torch.zeros(len(sents), seq_len)
        # dataX = torch.zeros(5, seq_len)
        dataY = []
        for label, sent in sents:
            review_text = BeautifulSoup(sent).get_text()
            review_text = re.sub("[^a-zA-Z]"," ", review_text)
            words = word_tokenize(review_text.lower())  
            lemma_words = [lemmatizer.lemmatize(i) for i in words]
            dataX[idx][0] = SOS_token
            for j in range(1, seq_len):
                if j > len(lemma_words) + 1:
                    dataX[idx][j] = PAD_token
                elif j == len(lemma_words) + 1 or j == seq_len-1:
                    dataX[idx][j] = EOS_token
                elif lemma_words[j-1] in word_to_ix:
                    dataX[idx][j] = word_to_ix[lemma_words[j-1]]
                else:
                    dataX[idx][j] = UNK_token
            dataY.append(label)
            idx += 1
        return dataX.long(), dataY

    def __len__(self):
        # return 5
        return len(self.dataX)
    
    def __getitem__(self, id):
        return self.dataX[id], self.dataY[id]
