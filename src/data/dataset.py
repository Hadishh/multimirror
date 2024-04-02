from transformers import BertTokenizer
import csv
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

csv.field_size_limit(1_000_000)

class AlignmentDataset(Dataset):
    __tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    @staticmethod
    def collate_fn(batch):
        SEP = AlignmentDataset.__tokenizer.sep_token
        PAD = AlignmentDataset.__tokenizer.pad_token
        pairs = [ 
                  item[0][0] + f" {SEP} " \
                  + item[0][1] \
                for item in batch
                ]
        bert_input = AlignmentDataset.__tokenizer(pairs, return_tensors="pt", padding="longest")
        labels = [item[1] for item in batch]
        #build indexes
        indices = list()
        for i in range(len(batch)):
            sentence = pairs[i].split()
            idx = 1
            indices_ = list()
            temp_idx = [0]
            for j, word in enumerate(sentence):
                if word == SEP:
                    idx = 0
                    indices_.append(temp_idx)
                    temp_idx = []
                
                tokens = AlignmentDataset.__tokenizer.tokenize(word)
                temp_idx.extend([idx] * len(tokens))
                idx += 1
            indices_.append(temp_idx)
            indices.append(indices_)

        return bert_input, labels, indices
        
    def __init__(self, path_to_tsv):
        self.pairs = list()
        self.labels = list()

        with open(path_to_tsv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n").split("\t")
                self.__parse_instance(line)
    
    def __parse_instance(self, line):
        if len(line) < 3:
            s1, s2 = tuple(line)
            self.labels.append(None)
        else:
            s1, s2, alignments = tuple(line)
            #get sentence lengths
            l = len(s1.split())
            k = len(s2.split())
            
            #parse alignments
            alignments = alignments.split()
            alignments = [(int(a.split('-')[0]), int(a.split('-')[1])) for a in alignments]

            #produce labels tensor
            label = torch.zeros(l, k)
            for u, v in alignments:
                label[u, v] = 1
            
            #append to data
            self.labels.append(label)
        self.pairs.append((s1, s2))
    
    def __getitem__(self, index) -> tuple():
        return self.pairs[index], self.labels[index]
    
    def __len__(self):
        return len(self.pairs)