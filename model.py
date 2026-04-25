import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, dim_model: int):
        super().__init__()
        self.dim_model=dim_model
        self.vocab_size=vocab_size
        self.embeddings=nn.Embedding(vocab_size,dim_model)

    def forward(self,x):
        return self.embeddings(x)*math.sqrt(self.dim_model)


class PositionalEmbeddings(nn.Module):
    def __init__(self, dim_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dim_model = dim_model

    def forward(self, x):
        seq_len = x.size(1)
        pe = torch.zeros(seq_len, self.dim_model, device=x.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim_model, 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / self.dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        x = x + pe.requires_grad_(False)
        return self.dropout(x)

class LayerNormalisation(nn.Module):
    def __init__(self , dim_model: int = 512 , eps: float = 1e-6):
        super().__init__()
        self.eps=eps
        self.dim_model=dim_model
        self.gamma = nn.Parameter(torch.ones(512)) #Multiplied
        self.beta = nn.Parameter(torch.zeros(512)) #Added
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True,unbiased=False)
        return self.gamma * (x-mean)/(std+self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, dim_model: int , dropout: float, dim_ff: int=2048):
        super().__init__()
        self.linear1=nn.Linear(dim_model,dim_ff) #Linear Layer 1 with shape (dim_model, dim_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(dim_ff,dim_model) #Output Layer with shape (dim_ff, dim_model)

    def forward(self,x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = dim_model // num_heads

        self.query = nn.Linear(dim_model, dim_model)
        self.key = nn.Linear(dim_model, dim_model)
        self.value = nn.Linear(dim_model, dim_model)
        self.output = nn.Linear(dim_model, dim_model)
    
    def attention(self, query, key, value, mask):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        query = query.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        x, self.attention_scores = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.head_dim)

        return self.output(x)
    
class ResidualConnection(nn.Module):
    def __init__(self,dropout: int =0.1, dim_model: int = 512 ):
        super().__init__()
        self.norm=LayerNormalisation(dim_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,layer):
        return x+self.dropout(layer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, attention_block: MultiHeadAttention , feed_forward_block: FeedForwardBlock, residual_connections: ResidualConnection,  dropout: float):
        super().__init__()
        self.attention_block=attention_block
        self.feedforward_block=feed_forward_block
        self.residual_connection1=residual_connections
        self.residual_connection2=residual_connections

    def forward(self,x,src_mask):
        x=self.residual_connection1(x, lambda x: self.attention_block(x,x,x,src_mask))
        x=self.residual_connection2(x,self.feedforward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList,dim_model: int=512):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation(dim_model)
    
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention,feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_Attention_block =self_attention_block
        self.cross_Attention_block=cross_attention_block
        self.feedforward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x,lambda x: self.self_Attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x: self.cross_Attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feedforward_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList,dim_model: int=512):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation(dim_model)
    
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, dim_model: int, vocab_size: int):
        super().__init__()
        self.linear=nn.Linear(dim_model,vocab_size)
    
    def forward(self,x):
        return F.log_softmax(self.linear(x),dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embeddings: InputEmbeddings, tgt_embeddings: InputEmbeddings, src_pos: PositionalEmbeddings, tgt_pos: PositionalEmbeddings,projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embeddings=src_embeddings
        self.tgt_embeddings=tgt_embeddings
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer
    
    def encode(self,src,src_mask):
        src=self.src_embeddings(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt=self.tgt_embeddings(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def projection(self,x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int , tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, dim_model: int = 512, N: int =6, num_heads: int = 8, dropout: float=0.1, dff: int =2048):
    #Create source and target embeddings
    src_embeddings=InputEmbeddings(src_vocab_size, dim_model)
    tgt_embeddings=InputEmbeddings(tgt_vocab_size, dim_model)

    #Create positional encoding layers
    src_pos=PositionalEmbeddings(dim_model,dropout)
    tgt_pos=PositionalEmbeddings(dim_model,dropout)

    #Create encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_Attention_block=MultiHeadAttention(dim_model,num_heads,dropout)
        feed_forward_block=FeedForwardBlock(dim_model,dropout,dff)
        residualconnection=ResidualConnection(dropout,dim_model)
        encoder_block=EncoderBlock(encoder_self_Attention_block,feed_forward_block,residualconnection,dropout=dropout)
        encoder_blocks.append(encoder_block)
    
    #Create decoder blocks
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_Attention_block=MultiHeadAttention(dim_model,num_heads,dropout)
        encoder_decoder_Attention_block=MultiHeadAttention(dim_model,num_heads,dropout)
        feed_forward_block=FeedForwardBlock(dim_model,dropout,dff)
        decoder_block=DecoderBlock(decoder_self_Attention_block,encoder_decoder_Attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)
    
    #Create encoder and decoder
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    #Create projection layer
    projection_layer=ProjectionLayer(dim_model,tgt_vocab_size)

    #Create the transformer
    transformer=Transformer(encoder,decoder,src_embeddings,tgt_embeddings,src_pos,tgt_pos,projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer




    


