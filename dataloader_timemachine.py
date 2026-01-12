import random
import torch
from pathlib import Path

from dataset_timemachine import *

class Dataloader_timemachine():
    def __init__(self,dataset,vocab,batch_size,num_steps,use_random_iter=True):

        self.dataset = dataset
        self.vocab = vocab
        self.corpus = [self.vocab[token] for line in self.dataset.tokenized_lines for token in line]
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.use_random_iter = use_random_iter

    def seq_iter_random(self):
        corpus = self.corpus[random.randint(0,self.num_steps-1):]
        num_subseqs = (len(corpus)-1) // self.num_steps
        init_idxs = list(range(0,num_subseqs * self.num_steps,self.num_steps))
        random.shuffle(init_idxs)

        num_batches = num_subseqs // self.batch_size
        for i in range(0,num_batches * self.batch_size,self.batch_size):
            idxs = init_idxs[i:i + self.batch_size]
            X = [corpus[j : j + self.num_steps] for j in idxs]
            Y = [corpus[j+1 : j + self.num_steps +1 ] for j in idxs]
            yield torch.tensor(X),torch.tensor(Y)

    def seq_iter_sequential(self): # 顺序分区,先计算token数,再reshape[batch_size,-1],再从第二维挑
        corpus = self.corpus[random.randint(0,self.num_steps-1):]
        num_tokens = (len(corpus)-1) // self.batch_size * self.batch_size
        Xs = torch.tensor(corpus[:num_tokens])
        Ys = torch.tensor(corpus[1:num_tokens+1])
        Xs,Ys = Xs.reshape(self.batch_size,-1),Ys.reshape(self.batch_size,-1)
        num_subseqs = Xs.shape[1]//self.num_steps
        for i in range(0,num_subseqs * self.num_steps,self.num_steps):
            X = Xs[:,i:i + self.num_steps]
            Y = Ys[:,i:i + self.num_steps]
            yield X,Y

    def __iter__(self):
        if self.use_random_iter == True:
            return self.seq_iter_random()
        else:
            return self.seq_iter_sequential()
        
if __name__ == "__main__":

    tmd = Dataset_timemachine()
    vocab = Vocab(tmd.tokenized_lines)

    dataloader = Dataloader_timemachine(tmd,vocab,batch_size=4,num_steps=20,use_random_iter=False)
    for X,Y in dataloader:
        X_tokens = [[vocab.idx2token[int(i)] for i in row] for row in X]
        for line in X_tokens:
            print(line)
        print("\n")
        Y_tokens = [[vocab.idx2token[int(i)] for i in row] for row in Y]
        for line in Y_tokens:
            print(line)
