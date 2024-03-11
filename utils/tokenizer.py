class CharTokenizer:
    def __init__(self, chars):
        self.ch_to_id = {ch:i for i, ch in enumerate(chars)}
        self.id_to_ch = {i:ch for i, ch in enumerate(chars)}
    
    def encode(self, s):
        return [self.ch_to_id[c] for c in s] # encoder: take a string, output a list of integers
    
    def decode(self, l):
        return ''.join([self.id_to_ch[i] for i in l]) # decoder: take a list of integers, output a string