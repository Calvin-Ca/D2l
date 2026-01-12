from dataset_timemachine import *
from dataloader_timemachine import *
import torch
from torch.nn import functional as F

class RNN_scratch():
    def __init__(self,
                 vocab_size,num_hiddens,    # 模型参数
                 device = "cpu",
                 ):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.device = device

        self.params = self._get_params()

    # 初始化隐状态
    def init_hidden_state(self,batch_size) -> tuple[torch.Tensor]:
        H_init = (torch.zeros((batch_size,self.num_hiddens),device=self.device),)
        return H_init
    
    # 初始化学习参数
    def _get_params(self):
        # 隐藏层,torch.randn生成指定形状的张量,张量的每个元素相互独立地从正态分布曲线随机获取
        W_xh = torch.randn(size = (self.vocab_size,self.num_hiddens),device=self.device) * 0.01 # # 防止爆炸
        W_hh = torch.randn(size = (self.num_hiddens,self.num_hiddens),device=self.device) * 0.01
        b_h = torch.zeros(self.num_hiddens,device=self.device)

        # 输出层
        W_q = torch.randn(size = (self.num_hiddens,self.vocab_size),device=self.device) * 0.01
        b_q = torch.zeros(self.vocab_size)

        # 附加梯度
        params = [W_xh,W_hh,b_h,W_q,b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def rnn(self,inputs,state): # inputs = (num_steps,batch_size,vocab_size)
        W_xh,W_hh,b_h,W_q,b_q = self.params
        H, = state
        out_puts = []
        for x in inputs:
            H = torch.mm(x,W_xh) + torch.mm(H,W_hh) + b_h
            H = torch.tanh(H)
            Y = torch.mm(H,W_q) + b_q
            out_puts.append(Y)
        Y = torch.cat(out_puts,dim=0)
        return Y,(H,)
    
    def predict(self,prefix,num_preds,vocab):
        # 初始化隐状态
        state = self.init_hidden_state(batch_size=1)
        
        outputs = [vocab[prefix[0]]]
        get_input = lambda:torch.tensor(outputs[-1],device=self.device).reshape((1,1))  # (batch_size,num_steps)
        # 预热期
        for y in prefix[1:]:
            _,state = self(get_input(),state)
            outputs.append(vocab[y])
        for _ in range(num_preds):
            y,state = self(get_input(),state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        result = [vocab.idx2token[i] for i in outputs]
        return ' '.join(result)
    
    def __call__(self,X, state): # X = (batch_size,num_steps)
        X = F.one_hot(X.T,num_classes=self.vocab_size).type(torch.float32).to(self.device)  # X = (num_steps,batch_size,vocab_size)
        return self.rnn(X,state)


if __name__ == "__main__":
    
    dataset = Dataset_timemachine()
    vocab = Vocab(dataset.tokenized_lines)
    dataloader = Dataloader_timemachine(dataset=dataset,vocab=vocab,batch_size=4,num_steps=5,use_random_iter=True)
    x,y = next(iter(dataloader))        # 访问单个样本
    
    X = torch.arange(10).reshape((2,5)) # 输入,(batch_size,num_steps)
    net = RNN_scratch(vocab_size=len(vocab),num_hiddens=512,device="cpu")   # 模型
    Y,state = net(X=X,state=net.init_hidden_state(batch_size=X.shape[0]))   # 推理
    predict_result = net.predict(prefix="time traveller",num_preds=10,vocab=vocab)
    print(predict_result)
