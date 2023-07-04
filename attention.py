import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale # 缩放量 \sqrt(dim)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, q, k, v, mask=None):
        # 1. DotProduct
        a = torch.bmm(q, k.transpose(2, 1)) # B T_q H x B H T_k -> B T_q T_k

        # 2. Scale
        a = a / self.scale

        # 3. Mask
        if mask is not None:
            a = a.masked_fill(mask, -np.inf)
        
        # 4. SoftMax
        a = self.softmax(a) 

        # 5. Output
        output = torch.bmm(a, v) # B T_q T_k x B T_v H_v (T_v == T_k) -> B T_q H_v
        return a, output

class SelfAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.q_W = nn.Linear(query_dim, query_dim)
        self.k_W = nn.Linear(key_dim, key_dim, bias=False)
        self.v_W = nn.Linear(value_dim, value_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, mask):
        query = self.q_W(query)
        key = self.k_W(key)
        value = self.v_W(value)

        scale = np.sqrt(query.shape[-1])
        attn = torch.bmm(query, key.transpose(2, 1)) / scale

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        
        attn = self.softmax(attn)
        output = torch.bmm(self.dropout(attn), value)
        return attn, output


class MultiHeadAttention(nn.Module):
    ''' Multi-head attention allows the model to jointly attend to 
        information from different representation subspaces at different positions.
    '''
    def __init__(self, head, d_model, dropout=0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.q_W = nn.Linear(d_model, d_model)
        self.k_W = nn.Linear(d_model, d_model, bias=False)
        self.v_W = nn.Linear(d_model, d_model)
        self.o_W = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        assert (d_model % head == 0)
        self.num_heads = head
        self.head_dim = d_model // head
        self.scale = self.head_dim**-0.5
        self.d_model = d_model
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, past_key_value=None, mask=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_W(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_W(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_W(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2] # q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(self.head_dim)

        
        if mask is not None:
            attn_weights = attn_weights + mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
        attn_weights = self.softmax(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.d_model)

        attn_output = self.o_W(attn_output)
        return attn_output, attn_weights, past_key_value
        # if mask is not None:
        #     attn_weights.masked_fill_(mask)
        



        
        

if __name__ == "__main__":
    # batch = 2
    # n_q, n_k, n_v = 2, 4, 4
    # d_q, d_k, d_v = 128, 128, 64

    # q = torch.randn(batch, n_q, d_q)
    # k = torch.randn(batch, n_k, d_k)
    # v = torch.randn(batch, n_v, d_v)
    # mask = torch.zeros(batch, n_q, n_k).bool()
    # mask[0,-1,2:] = True
    # # print(mask)
    # attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
    # attn, output = attention(q, k, v, mask=mask)
    # print(attn.shape)
    # print(output.shape)

    batch = 2
    n_q, n_k, n_v = 4, 4, 4
    d_q, d_k, d_v = 256, 256, 256

    q = torch.randn(batch, n_q, d_q)
    # k = torch.randn(batch, n_k, d_k)
    # v = torch.randn(batch, n_v, d_v)
    mask = torch.zeros(batch, 1, n_q, n_k)
    mask[0,0,-1,2:] = -np.inf
    # print(mask)
  
    mha = MultiHeadAttention(8, 256)
 
    output, attn, past_kv = mha(q, mask=mask)
  
    print(attn.shape)
    print(attn[0][0])
    print(output.shape)

    # x = torch.randn(2, 12, 256)
    # new_x_shape = (2, 12) + (8, 32)
    # x = x.view(new_x_shape)
    # print(x.permute(0, 2, 1, 3).size()) # bsz num_heads seq_len head_dim