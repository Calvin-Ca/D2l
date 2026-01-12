from pathlib import Path
import re
import requests 
import collections

class Dataset_timemachine:
    def __init__(self,
                 root=Path('.').parent/'dataset/timemachine',
                 tokenize_mode="word",
                 ):
        self.root = Path(root) if Path(root).exists else Path(root).mkdir(parents=True,exist_ok=True)
        self.tokenize_mode = tokenize_mode
        self.cleaned_lines = None
        self.tokenized_lines = None

        self._prepare()

    def _prepare(self):
        self.download_time_machine()
        self.clean()
        self.tokenize()

    def download_time_machine(self):
        if not self.root.exists():
            url = "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
            response = requests.get(url) 
            response.raise_for_status()
            with open(self.root,'wb') as f:
                f.write(response.content)
            print(f"time_machine downloaded at {self.root}")
    
    def clean(self):
        with open(self.root) as f:
            lines = f.readlines()
            lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
        self.cleaned_lines = lines
    
    def tokenize(self):
        if self.tokenize_mode == 'char':
            lines = [list(line) for line in self.cleaned_lines]
        elif self.tokenize_mode == 'word':
            lines = [line.split() for line in self.cleaned_lines]
        else:
            ValueError
        self.tokenized_lines  = lines
    
class Vocab():
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if tokens == None:
            tokens = []
        if reserved_tokens == None:
            reserved_tokens = []

        tokens_counter = self._count_corus(tokens)
        self.token_freq = sorted(tokens_counter.items(),key=lambda x:x[1],reverse=True) # self.token_freq,list

        self.idx2token = reserved_tokens + ['<unk>']
        self.token2idx = {token:idx for idx,token in enumerate(self.idx2token)}

        for token,freq in self.token_freq:
            if freq < min_freq:
                break
            else:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token)-1

    def __len__(self):
        return len(self.token2idx)
    
    def __getitem__(self,tokens):
        if not isinstance(tokens,(list,tuple)):                 # 单个token,字符串格式 
            return self.token2idx.get(tokens,self.unk)
        return [self.__getitem__(token for token in tokens)]    # token的list或tuple
    
    def _count_corus(self,tokens):
        if len(tokens) == 0 or isinstance(tokens[0],list):      # 短路机制,避免传入[]使得token[0]报错
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property      # 允许 obj.x
    def unk(self):
            return 0

    @property
    def token_freqs(self):
            return self._token_freqs

if __name__ == "__main__":
    dateset_root = Path('.').parent/'dataset/timemachine'
    tmd = Dataset_timemachine(root=dateset_root,tokenize_mode='word')
    vocab = Vocab(tmd.tokenized_lines)
    vocab['me']
    pass
        


