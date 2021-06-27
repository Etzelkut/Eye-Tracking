from basic_dependency import *
from functools import partial
from local_attention import LocalAttention
from functional import generate_relative_positions_matrix, default, split_at_index

#copied from https://fast-transformers.github.io/attention/
#https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/attention_layer.py

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.
    Given the queries, keys and values as Q, K, V instead of computing
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    we make use of a feature map function Φ(.) and perform the following
    computation
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, feature_map=None, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = feature_map or elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        # Apply the feature map to the queries and keys
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        #K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries):
        """
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Project the queries/keys/values
        queries = self.query_projection(queries)
        keys = self.key_projection(queries)
        values = self.value_projection(queries)

        # Reshape them into many heads and compute the attention
        N, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        new_values = self.inner_attention(
            queries.view(N, L, H, -1),
            keys.view(N, S, H, -1),
            values.view(N, S, H, -1),
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


#https://github.com/CyberZHG/torch-multi-head-attention/blob/66f6ae801a6d2aea8994ef00af06fdfc67ec2026/torch_multi_head_attention/multi_head_attention.py#L77
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):

      batch_size, head_num, seq_len, sub_dim = query.shape

      query = query.reshape(batch_size * head_num, seq_len, sub_dim)
      key = key.reshape(batch_size * head_num, seq_len, sub_dim)
      value = value.reshape(batch_size * head_num, seq_len, sub_dim)

      dk = query.size()[-1]
      scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
      if mask is not None:
          scores = scores.masked_fill(mask == 0, -1e9)
      attention = F.softmax(scores, dim=-1)
      return attention.matmul(value).reshape(batch_size, head_num, seq_len, sub_dim)



#https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py#L143

class SelfAttention_local(nn.Module):
    def __init__(self, dim, heads, dim_head = None, dropout = 0., attn_dropout = 0.,
                 n_local_attn_heads = 0, local_attn_window_size = 128,):
        super().__init__()
        causal = False
        one_kv_head = False
        receives_context = False

        assert dim_head or (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        d_heads = default(dim_head, dim // heads)

        self.heads = heads
        self.d_heads = d_heads
        self.receives_context = receives_context

        self.global_attn_heads = heads - n_local_attn_heads
        self.global_attn_fn = ScaledDotProductAttention() #linear_attn #if not causal else partial(causal_linear_attn, bucket_size = blindspot_size)

        self.local_attn_heads = n_local_attn_heads
        self.local_attn  = LocalAttention(local_attn_window_size, causal = causal, dropout = attn_dropout)

        self.to_q = nn.Linear(dim, d_heads * heads, bias = False)

        kv_heads = (int(self.local_attn_heads > 0) + int(self.global_attn_heads > 0)) if one_kv_head else heads

        self.one_kv_head = one_kv_head
        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias = False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias = False)

        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask = None, context = None, context_mask = None, **kwargs):
        assert not (self.receives_context and context is None), 'context must be supplied if self attention is in receives context mode'

        if not self.receives_context:
            q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        else:
            q, k, v = (self.to_q(x), self.to_k(context), self.to_v(context))

        b, t, e, h, dh = *q.shape, self.heads, self.d_heads

        merge_heads = lambda x: x.reshape(*x.shape[:2], -1, dh).transpose(1, 2)

        q, k, v = map(merge_heads, (q, k, v))

        out = []

        split_index_fn = partial(split_at_index, 1, self.local_attn_heads)
        if not self.one_kv_head:
            (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
        else:
            lq, q = split_index_fn(q)

            split_kv_fn = partial(split_at_index, 1, int(self.local_attn_heads > 0))
            (lk, k), (lv, v) = map(split_kv_fn, (k, v))

            local_expand_heads_fn = lambda t: expand_dim(t, 1, self.local_attn_heads, unsqueeze=False)
            lk, lv = map(local_expand_heads_fn, (lk, lv))

            k, v = map(lambda t: t.squeeze(1), (k, v))

        has_local, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

        if has_local:
            local_out = self.local_attn(lq, lk, lv, input_mask = input_mask)
            out.append(local_out)

        if has_global:
            kv_mask = input_mask if not self.receives_context else context_mask

            global_out = self.global_attn_fn(q, k, v)#self.global_attn_fn(q, k, v, one_kv_head = self.one_kv_head, kv_mask = kv_mask)
            out.append(global_out)

        attn = torch.cat(out, dim=1)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.dropout(self.to_out(attn))



class MultiheadAttentionRPR(nn.Module):
    def __init__(self, embed_dim, num_heads, max_relative_positions, dropout=0.):
        super(MultiheadAttentionRPR, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)
        self._reset_parameters()
        self.max_relative_positions = max_relative_positions
        vocab_size = max_relative_positions * 2 + 1
        self.relative_positions_embeddings = nn.Embedding(vocab_size, self.head_dim)

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query, need_weights=True):

            return multi_head_attention_forwardRPR(self.max_relative_positions, self.relative_positions_embeddings,
                query, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,  self.dropout, 
                self.out_proj.weight, self.out_proj.bias, 
                training=self.training, need_weights=need_weights)

def multi_head_attention_forwardRPR(max_relative_positions, relative_positions_embeddings, 
                                query, num_heads,                       
                                 in_proj_weight, in_proj_bias, dropout_p,                       
                                 out_proj_weight, out_proj_bias,        # out_proj
                                 training=True, need_weights=True):
    tgt_len, bsz, embed_dim = query.size()
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5

    q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    relative_positions_matrix = generate_relative_positions_matrix(
        tgt_len, max_relative_positions,
        cache=False)

    relations_keys = relative_positions_embeddings(
        relative_positions_matrix.to(query.device))

    relations_values = relative_positions_embeddings(
        relative_positions_matrix.to(query.device))    

    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    rel_keys = torch.bmm(q.transpose(0, 1), relations_keys.transpose(1, 2)).transpose(0,1)
    attn_output_weights = attn_output_weights + rel_keys


    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    rel_values = torch.bmm(attn_output_weights.transpose(0, 1), relations_values).transpose(0,1)
    attn_output = attn_output + rel_values
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
