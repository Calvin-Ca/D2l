import torch
from d2l import torch as d2l
import math
from torch import nn
from rnn import *
from dataset_timemachine import *
from dataloader_timemachine import *

import torch
from d2l import torch as d2l
import math
from torch import nn

import time

# tools
class Timer():
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.times.append(time.time()-self.start_time)
        return self.times[-1]
    
class Accumulator():
    def __init__(self,n):
        self.data = [0.0] * n
    
    def __getitem__(self,idx):
        return self.data[idx]
    
    def add(self,*args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]

class RNNTrainer_scratch:
    def __init__(self, dataloader, net, lr, vocab, device="cpu"):
        self.dataloader = dataloader
        self.net = net
        self.vocab = vocab
        self.device = device
        self.lr = lr
        self.loss = nn.CrossEntropyLoss() # (batch_size,Class)未经过 Softmax 的原始得分;(N)真实类别索引（LongTensor）
        self.optimizer = self._sgd

    def _sgd(self,batch_size):
        with torch.no_grad():
            for p in self.net.params:
                p -= self.lr * p.grad / batch_size      
                p.grad.zero_() 

    def _grad_clip(self, theta):
        global_param_grad2_sum = torch.tensor([0.0], device=self.device)
        for p in self.params:
            if p.grad is not None:
                global_param_grad2_sum += torch.sum(p.grad ** 2)
        norm = torch.sqrt(global_param_grad2_sum)

        if norm > theta:
            for p in self.params:
                if p.grad is not None:
                    p.grad[:] *= theta / norm

    def train_epoch(self):
        state = None
        timer = Timer()
        metric = Accumulator(2)

        for X, Y in self.dataloader:
            if state is None or self.dataloader.use_random_iter:            # 相邻batch之间无关
                state = self.net.init_hidden_state(batch_size=X.shape[0])
            else:
                if isinstance(state, (list, tuple)):
                    for s in state: 
                        s.detach_()
                else:
                    state.detach_()
            X, y = X.to(self.device),Y.T.reshape(-1).to(self.device).long()
            y_hat, state = self.net(X, state)
            l = self.loss(y_hat, y)

            # 用来从某个标量开始,沿着计算图反向传播,计算所有requires_grad=True参数的梯度,并把结果累加到.grad中
            l.backward()
            self._grad_clip(theta=1)
            self.optimizer(batch_size=1)
            metric.add(l * y.numel(), y.numel())
            
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
      
    def fit(self, num_epochs):
        # 可视化,xlim 为x轴显示范围,从第10个epoch 开始画
        animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
        
        for epoch in range(num_epochs):
            ppl, speed = self.train_epoch()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: {self.net.predict('time traveller', 50, self.vocab)}")
                animator.add(epoch + 1, [ppl])
        
        print(f'最终困惑度: {ppl:.1f}, 速度: {speed:.1f} 词元/秒')

if __name__ == "__main__":
    dataset = Dataset_timemachine()
    vocab = Vocab(dataset.tokenized_lines)
    dataloader = Dataloader_timemachine(dataset=dataset,vocab=vocab,batch_size=4,num_steps=5,use_random_iter=False)
    num_epochs,lr = 10,1
    net = RNN_scratch(
        vocab_size=len(vocab),
        num_hiddens=512,
        device="cpu",
        )
    rnn_trainer = RNNTrainer_scratch(dataloader, net, lr, vocab, device="cpu")
    rnn_trainer.fit(num_epochs=num_epochs)
