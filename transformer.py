import torch
import torch.nn.functional as F
import math
import torch.nn as nn

class MSA:
    "q:(batch_size,len_seq_q,embed_dim)" 
    "k:(batch_size,len_seq_k,kdim)"
    "v:(batch_size,len_seq_k,vdim)"
    "self.head_dim = embed_dim // num_head"
    def __init__(self,embed_dim=64,num_heads=1,kdim=256,vdim=256) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim
        # 动机：将embed_dim拆成num_head个head_dim维的子空间,复杂问题简单化,并行处理多维度的语义
        self.head_dim = embed_dim // num_heads  
        
        
        self.Wq = torch.randn((embed_dim,embed_dim)) / math.sqrt(embed_dim)
        self.Wk = torch.randn(kdim,embed_dim)  / math.sqrt(embed_dim)
        self.Wv= torch.randn(vdim,embed_dim)  / math.sqrt(embed_dim)
        # 为了保证层与层之间的流水线对齐,
        # 例如Transformer 每一层都有一个公式：Output = Layer(x) + x,输出维度与输入维度要一致
        self.wo = torch.randn(embed_dim,embed_dim)  / math.sqrt(embed_dim)

    def __call__(self,query,key,value):
        return self.forward(query,key,value)
    
    def forward(self,query,key,value,attn_mask=None):
        batch_size,len_seq_q,embed_dim = query.shape
        len_seq_k = key.shape[1]

        Q = torch.matmul(query,self.Wq) # (batch_size,len_seq_q,embed_dim) = (batch_size,len_seq_q,embed_dim) @ (batch_size,embed_dim,embed_dim)
        K = torch.matmul(key,self.Wk)   # (batch_size,len_seq_k,embed_dim) = (batch_size,len_seq_k,kdim) @ (batch_size,kdim,embed_dim)
        V = torch.matmul(value,self.Wv) # (batch_size,len_seq_k,embed_dim) = (batch_size,len_seq_k,vdim) @ (batch_size,vim,embed_dim)

        # 用不同的 head 关注 seq 不同角度的特征,(batch_size,num_heads,len_seq(q/k),head_dim)
        Q = Q.view(batch_size,len_seq_q,self.num_heads,self.head_dim).transpose(1,2) 
        K = K.view(batch_size,len_seq_k,self.num_heads,self.head_dim).transpose(1,2)
        V = V.view(batch_size,len_seq_k,self.num_heads,self.head_dim).transpose(1,2)

        # (batch_size,num_heads,len_seq_q,len_seq_k)
        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0,float('-inf'))
        attn_weights = F.softmax(scores,dim=-1)

        # (batch_size,num_heads,len_seq_q,head_dim) = /
        # (batch_size,num_heads,len_seq_q,len_seq_k) @ (batch_size,num_heads,len_seq_k,head_dim)
        context = torch.matmul(attn_weights,V)

        # (batch_size,len_seq_q,embed_dim)
        context = context.transpose(1,2).contiguous().view(batch_size,len_seq_q,self.embed_dim)

        # (batch_size,len_seq_q,embed_dim) @ (embed_dim,embed_dim)
        out = torch.matmul(context,self.wo)
        
        return out,attn_weights


def positional_encoding(seq_len, d_model):
    """
    seq_len: 序列长度 (例如 100)
    d_model: 特征维度 (例如 64)
    """
    # 1. 初始化矩阵 (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    
    # 2. (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    # 3. 计算分母项 (10000^(2i/d_model)),这里使用对数空间计算，防止数值溢出
    # (d_model/2)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    # 4. 填充矩阵：偶数维度用sin,奇数维度用cos
    ### broadcast mechanism
    # (seq_len, 1)*(d_model/2)
    # ->(seq_len, 1)*(1,d_model/2)
    # ->(seq_len, d_model/2)*(1,d_model/2)
    # ->(seq_len, d_model/2)*(seq_len,d_model/2)
    # (seq_len,d_model/2)
    pe[:, 0: d_model :2] = torch.sin(position * div_term) 
    pe[:, 1: d_model :2] = torch.cos(position * div_term)
    
    return pe


if __name__ == "__main__":

    mha = nn.MultiheadAttention(embed_dim=1028,num_heads=1,kdim=256,vdim=256,batch_first=True)
    query,key,value = torch.randn(1, 5, 1028),torch.randn(1, 10, 256),torch.randn(1, 10, 256)
    out, _ = mha(query, key, value)

    pe_matrix = positional_encoding(seq_len=50, d_model=64)
    print(f"{pe_matrix.shape}") # (50, 64)



