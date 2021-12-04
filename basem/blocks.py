from basem.basic_dependency import *
from basem.subblocks import PositionalEncoding, AxialPositionalEncoding, GLUblock, GLUblock_no_wn, conv3x3, conv1x1, GLU_alt
from basem.subblocks import Branched_conv_2d, Branched_conv_stand, Group_End_1d_Wrap, Group_End_2d_Wrap
from basem.attn import LinearAttention, MultiheadAttentionRPR, AttentionLayer, SelfAttention_local
from basem.functional import mish_f

#import copy

class MultiheadAttention_wrap(nn.Module):
    def __init__(self, d, num_heads, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(d, num_heads, dropout)
    
    def forward(self, x):
        x = x.transpose(0,1)#.contiguous()
        x, _= self.att(x, x, x)
        x = x.transpose(0,1)#.contiguous()
        return x



class GLUblock_wrap(nn.Module):
    def __init__(self, typee, k, in_c, out_c, downbot):
        super().__init__()
        if typee == "gcn":
            self.att = GLUblock(k, in_c, out_c, downbot)
        else:
            self.att = GLUblock_no_wn(k, in_c, out_c, downbot)
    
    def forward(self, x):
        x.unsqueeze_(-1)
        x = x.transpose(1,2)#.contiguous()
        x = self.att(x)
        x = x.transpose(1,2)#.contiguous()
        x.squeeze_(-1)

        return x


class MultiheadAttentionRPR_wrap(nn.Module):
    def __init__(self, d, num_heads, max_relative_positions_rpr, dropout):
        super().__init__()    
        self.att = nn.MultiheadAttentionRPR(d, num_heads, max_relative_positions_rpr, dropout)
    
    def forward(self, x):
        x = x.transpose(0,1)#.contiguous()
        x, _ = self.att(x)
        x = x.transpose(0,1)#.contiguous()
        return x


class Attention_Block(nn.Module):
  def __init__(self, type_dict):
    super().__init__()
    print("used type is: ", type_dict)
    choose_module = {
      #"dynamic": Dynamic_module(type_module = type_dict["type_module"], 
      #                             d = type_dict["d"], kernel_size = type_dict["kernel_size_dynamic"], 
      #                             num_heads = type_dict["num_heads"], dropout = type_dict["dropout"]),
                     
      #"light": Dynamic_module(type_module = type_dict["type_module"], 
      #                             d = type_dict["d"], kernel_size = type_dict["kernel_size_dynamic"], 
      #                             num_heads = type_dict["num_heads"], dropout = type_dict["dropout"]),

      "att" : MultiheadAttention_wrap(type_dict["d"], type_dict["num_heads"], type_dict["dropout"]),
      
      "gcn" : GLUblock_wrap("gcn", type_dict["k_kernel_glu"], type_dict["d"], type_dict["d"], type_dict["downbot_glu"]),
      "gcn_no_wn" : GLUblock_wrap("gcn_no_wn", type_dict["k_kernel_glu"], type_dict["d"], type_dict["d"], type_dict["downbot_glu"]),
      
      "linear" : AttentionLayer(LinearAttention(), type_dict["d"], type_dict["num_heads"]),

      "rpr": MultiheadAttentionRPR_wrap(type_dict["d"], type_dict["num_heads"], type_dict["max_relative_positions_rpr"], type_dict["dropout"]),

      "local_sa": SelfAttention_local(type_dict["d"], type_dict["num_heads"], 
                                n_local_attn_heads = type_dict["n_local_attn_heads"], dropout = type_dict["dropout"],
                                local_attn_window_size = type_dict["local_attn_window_size"]),
      "glu_alt": GLU_alt(type_dict["d"], num_layers=type_dict["alt_num_layers"], patch_size=type_dict["alt_patch_size"], padding=type_dict["alt_padding"]),                          
    }
    self.type_module = type_dict["type_module"]
    self.block = choose_module[type_dict["type_module"]]
  
  def forward(self, x):
    x = self.block(x)
    return x



class Resize_1dconv_wrap(nn.Module):
    def __init__(self,  size, new_size):
        super().__init__()
        self.resize = nn.Conv1d(size, new_size, 1)
    def forward(self, x):
        x = x.transpose(1,2)#.contiguous()
        x = self.resize(x)
        x = x.transpose(1,2)#.contiguous()
        return x



class Resize_Module(nn.Module):
  def __init__(self, type_module, size, new_size):
    super().__init__()
    print("resize set to: ", size, " -> ", new_size)
    if type_module == "fc":
      self.resize = nn.Linear(in_features = size, out_features = new_size)
    elif type_module == "1dconv":
      self.resize = Resize_1dconv_wrap(size, new_size)
    else:
      raise Exception("Sorry, no such method")

  def forward(self, x):
    x = self.resize(x)
    return x



class Pos_Emb(nn.Module):
  def __init__(self, type_module, d, dropout):
    super(Pos_Emb, self).__init__()
    if type_module == "const":
      self.pos_emb = PositionalEncoding(d, dropout)
    elif type_module == "axial":
      self.pos_emb = AxialPositionalEncoding(d, dropout)
    else:
      self.pos_emb = nn.Identity()
    
  def forward(self, x):
    return self.pos_emb(x)


class Group_End_Module(nn.Module):
  def __init__(self, type_module, d_model):
    super().__init__()

    if type_module == "2d":
      self.groupend = Group_End_2d_Wrap(d_model)

    elif type_module == "1d":
      self.groupend = Group_End_1d_Wrap(d_model)

    else:
      raise Exception("Sorry, no such method")

  def forward(self, x):
    x = self.groupend(x)
    return x


class Branched_Module(nn.Module):
  def __init__(self, type_module, d_model, activation = mish_f):
    super().__init__()

    if type_module == "2d":
      self.branches = Branched_conv_2d(d_model, activation)
    elif type_module == "1d":
      self.branches = Branched_conv_stand(d_model, activation)
    else:
      raise Exception("Sorry, no such method")

  def forward(self, x):
    x = self.branches(x)
    return x



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hid = None, activation = F.relu, glu = False, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        print("print positionwise Feed Forward with glu: ", glu)
        if d_hid is None:
          d_hid = d_model * 4
        d_hid_ = d_hid * (2 if glu else 1)

        self.glu = glu

        self.w_1 = nn.Linear(d_model, d_hid_)
        self.activation = activation
        self.w_2 = nn.Linear(d_hid, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        
        output = x
        if not self.glu:
          output = self.activation(self.w_1(output))
        else:
          output, gating = self.w_1(output).chunk(2, dim=-1)
          output = self.activation(output) * gating
  
        output = self.w_2(output)

        output = self.dropout(output)
        output = self.layer_norm(output + x) #residual

        return output



#https://github.com/xcmyz/FastSpeech/blob/896fe0276840267ba9565f4d620673695f7eef06/transformer/SubLayers.py#L72
class PositionwiseFeedForward_conv(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid = None, glu = False, activation = F.relu, dropout=0.1):
        super().__init__()
        print("print conv Feed Forward with glu: ", glu)
        if d_hid is None:
          d_hid = d_in * 4
        d_hid_ = d_hid * (2 if glu else 1)
        self.glu = glu
        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid_, kernel_size=9, padding=4)
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=3, padding=1)

        self.activation = activation

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #residual = x
        output = x.transpose(1, 2)
        if not self.glu:
          output = self.activation(self.w_1(output))
        else:
          output, gating = self.w_1(output).transpose(-1, -2).chunk(2, dim=-1)
          output = self.activation(output) * gating
          output = output.transpose(-1, -2)
  
        output = self.w_2(output)
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + x) #residual

        return output




class Conv2Block(nn.Module):
  def __init__(self, inn, out):
    super(Conv2Block, self).__init__()
    norm_layer = nn.BatchNorm2d

    self.conv1 = conv3x3(inn, out)
    self.bn1 = norm_layer(out)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(out, out)
    self.bn2 = norm_layer(out)

  def forward(self, x):
      identity = torch.mean(x, dim = 1)[:, None]

      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)

      out += identity
      
      out = self.relu(out)

      return out