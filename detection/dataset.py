import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from embedding.koreanphoneme import construct_vocab



class BERTDataset(Dataset):
    def __init__(self, dataset_path, epoch,
                 sow_token, eow_token, max_char_len, max_word_len):
        
        self.vocab = construct_vocab()
        self.sow_token, self.eow_token = sow_token, eow_token
        self.sow_idx, self.eow_idx     = self.vocab['<SOW>'], self.vocab['<EOW>']
        self.pad_idx, self.unk_idx     = self.vocab['<PAD>'], self.vocab['<UNK>']
        self.max_char_len = max_char_len
        self.max_word_len = max_word_len

        s_time = time.time()    
        with open(dataset_path+f'total_{epoch}.json', 'r') as f:
            raw_dataset = json.load(f)
        print(f'Epoch {epoch} Dataset Loaded! {time.time() - s_time}s')

        self.dataset = list(zip([d[1] for d in raw_dataset], 
                                [d[0] for d in raw_dataset]))[:1000]

        
    def sent2idx(self, sent):
        """
            Input form: 장르로는 판타지
                        -> indices of [[장, 르, 로, 는], [판, 타, 지]]
        """
        words = sent.split(' ')
        word_pad_indices = np.full(self.max_char_len, self.pad_idx)

        indices = []
        for word in words:
            indices_ = [self.vocab.get(char, self.unk_idx) for char in word]
            if self.sow_token: indices_.insert(0, self.sow_idx)
            if self.eow_token: indices_.append(self.eow_idx)

            indices_ = indices_[:self.max_char_len]
            indices_ = np.pad(indices_, (0, self.max_char_len - len(indices_)), constant_values=self.pad_idx)
            indices.append(indices_)

        indices.extend([word_pad_indices] * (self.max_word_len - len(indices)))
        return np.array(indices)
        
        
    def make_binary_label(self, sent1, sent2):
        """
            Output form: 장르로는 판타지 => 장루로는 판타지
                         -> [1, 0]
        """
        words1, words2 = sent1.split(' '), sent2.split(' ')      
        labels = np.array([1 if w1 != w2 else 0 for w1, w2 in zip(words1, words2)])
        labels = np.pad(labels, (0, self.max_word_len - len(labels)), constant_values=0)
        return labels
                                    
        
    def __len__(self):
        return len(self.dataset)
        
        
    def __getitem__(self, idx):
        augmented, origin = self.dataset[idx][0], self.dataset[idx][1]
        input  = self.sent2idx(augmented)
        label  = self.make_binary_label(augmented, origin)

        output = {'input': input, 'label': label}
        return {k: torch.tensor(v) for k, v in output.items()}