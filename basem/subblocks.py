from basem.basic_dependency import *
from axial_positional_embedding import AxialPositionalEmbedding
#from fairseq.modules import LightweightConv, DynamicConv


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


#copied
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=600):
        super(PositionalEncoding, self).__init__()
        print("pos encoder static")
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return x


class AxialPositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, dim, dropout, max_len=32): #32*32 = 1024 > 500
        super(AxialPositionalEncoding, self).__init__()
        print("axial pos encoder")
        if dim % 2 == 0:   
            self.pos_emb = AxialPositionalEmbedding(
                dim = dim,
                axial_shape = (max_len, max_len),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
                axial_dims = (int(dim/2), int(dim/2))          # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512)
            )
        else:
            self.pos_emb = AxialPositionalEmbedding(
                dim = dim,
                axial_shape = (max_len, max_len),          # axial shape will multiply up to the maximum sequence length allowed (64 * 64 = 4096)
            )            

    def forward(self, tokens):
        tokens = self.pos_emb(tokens) + tokens
        return tokens


#copied
class GLUblock(nn.Module):
    def __init__(self, k, in_c, out_c, downbot):
        super().__init__()
        #only need to change shape of the residual if num_channels changes (i.e. in_c != out_c)
        #[bs,in_c,seq_length]->conv(1,in_c,out_c)->[bs,out_c,seq_length]
        if in_c == out_c:
            self.use_proj=0
        else:
            self.use_proj=1
        self.convresid=nn.utils.weight_norm(nn.Conv2d(in_c, out_c, kernel_size=(1,1)),name='weight',dim=0)
        
        self.leftpad = nn.ConstantPad2d((0,0,k-1,0),0)#(paddingLeft, paddingRight, paddingTop, paddingBottom)

        #[bs,in_c,seq_length+(k-1)]->conv(1,in_c,in_c/downbot)->[bs,in_c/downbot,seq_length+(k-1)]
        self.convx1a = nn.utils.weight_norm(nn.Conv2d(in_c, int(in_c/downbot), kernel_size=(1,1)),name='weight',dim=0)
        self.convx2a = nn.utils.weight_norm(nn.Conv2d(in_c, int(in_c/downbot), kernel_size=(1,1)),name='weight',dim=0)
        #[bs,in_c/downbot,seq_length+(k-1)]->conv(k,in_c/downbot,in_c/downbot)->[bs,in_c/downbot,seq_length]
        self.convx1b = nn.utils.weight_norm(nn.Conv2d(int(in_c/downbot), int(in_c/downbot), kernel_size=(k,1)),name='weight',dim=0)
        self.convx2b = nn.utils.weight_norm(nn.Conv2d(int(in_c/downbot), int(in_c/downbot), kernel_size=(k,1)),name='weight',dim=0)
        #[bs,in_c/downbot,seq_length]->conv(1,in_c/downbot,out_c)->[bs,out_c,seq_length]
        self.convx1c = nn.utils.weight_norm(nn.Conv2d(int(in_c/downbot), out_c, kernel_size=(1,1)),name='weight',dim=0)
        self.convx2c = nn.utils.weight_norm(nn.Conv2d(int(in_c/downbot), out_c, kernel_size=(1,1)),name='weight',dim=0)
        self.active = Mish()
    def forward(self, x):
        residual = x
        if self.use_proj==1:# if in_c != out_c, need to change size of residual
            residual=self.convresid(residual)
        x=self.leftpad(x) # [bs,in_c,seq_length+(k-1),1]
        x1 = self.convx1c(self.convx1b(self.convx1a(x))) # [bs,out_c,seq_length,1]
        x2 = self.convx2c(self.convx2b(self.convx2a(x))) # [bs,out_c,seq_length,1]
        x2 = self.active(x2)#torch.sigmoid(x2)
        x=torch.mul(x1,x2) # [bs,out_c,seq_length,1]
        return x+residual



class GLUblock_no_wn(nn.Module):
    def __init__(self, k, in_c, out_c, downbot):
        super().__init__()
        #only need to change shape of the residual if num_channels changes (i.e. in_c != out_c)
        #[bs,in_c,seq_length]->conv(1,in_c,out_c)->[bs,out_c,seq_length]
        if in_c == out_c:
            self.use_proj=0
        else:
            self.use_proj=1
        self.convresid=nn.Conv2d(in_c, out_c, kernel_size=(1,1))
        
        self.leftpad = nn.ConstantPad2d((0,0,k-1,0),0)#(paddingLeft, paddingRight, paddingTop, paddingBottom)

        #[bs,in_c,seq_length+(k-1)]->conv(1,in_c,in_c/downbot)->[bs,in_c/downbot,seq_length+(k-1)]
        self.convx1a = nn.Conv2d(in_c, int(in_c/downbot), kernel_size=(1,1))
        self.convx2a = nn.Conv2d(in_c, int(in_c/downbot), kernel_size=(1,1))
        #[bs,in_c/downbot,seq_length+(k-1)]->conv(k,in_c/downbot,in_c/downbot)->[bs,in_c/downbot,seq_length]
        self.convx1b = nn.Conv2d(int(in_c/downbot), int(in_c/downbot), kernel_size=(k,1))
        self.convx2b = nn.Conv2d(int(in_c/downbot), int(in_c/downbot), kernel_size=(k,1))
        #[bs,in_c/downbot,seq_length]->conv(1,in_c/downbot,out_c)->[bs,out_c,seq_length]
        self.convx1c = nn.Conv2d(int(in_c/downbot), out_c, kernel_size=(1,1))
        self.convx2c = nn.Conv2d(int(in_c/downbot), out_c, kernel_size=(1,1))
        self.active = Mish()
    def forward(self, x):
        residual = x
        if self.use_proj==1:# if in_c != out_c, need to change size of residual
            residual=self.convresid(residual)
        x=self.leftpad(x) # [bs,in_c,seq_length+(k-1),1]
        x1 = self.convx1c(self.convx1b(self.convx1a(x))) # [bs,out_c,seq_length,1]
        x2 = self.convx2c(self.convx2b(self.convx2a(x))) # [bs,out_c,seq_length,1]
        x2 = self.active(x2)#torch.sigmoid(x2)
        x=torch.mul(x1,x2) # [bs,out_c,seq_length,1]
        return x+residual


"""
class Dynamic_module(nn.Module):
  def __init__(self, type_module, d, kernel_size, num_heads, dropout):
    super(Dynamic_module, self).__init__()
    #batch, input_w, input_dim
    self.d = d
    self.linear_w1 = nn.Linear(in_features = d, out_features = 2*d)
    self.active_glu = nn.Sigmoid()
    if type_module == "dynamic":
      self.lconv = DynamicConv(d, kernel_size = kernel_size, padding_l = 1, num_heads = num_heads, weight_dropout = dropout, 
                             weight_softmax = True, bias = True, conv_bias = True, in_proj = True)
    elif type_module == "light":
      self.lconv = LightweightConv(d, kernel_size = kernel_size, padding_l = 1, num_heads = num_heads, weight_dropout = dropout, 
                        weight_softmax = True, bias = True)
    self.linear_w0 = nn.Linear(in_features = d, out_features = d)
  
  def forward(self, X):
    X = self.linear_w1(X) #Q
    X = torch.mul(self.active_glu(X[:,:, :self.d]), X[:,:, self.d:]) #G
    X = X.transpose(0,1).contiguous() 
    X = self.lconv(X) #L
    X = X.transpose(0,1).contiguous()
    return self.linear_w0(X)

"""

# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



#####################################################


class GatedConvolution_alt(nn.Module):
    def __init__(self,d_model,patch_size=3,padding=1):
        super(GatedConvolution_alt,self).__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=2 * d_model,kernel_size=patch_size,padding=padding,bias=True)
        xavier_uniform_(self.conv.weight, gain=1)

    def forward(self,x):
        convoluted = self.conv(x.transpose(1,2)).transpose(1,2)
        out, gate = convoluted.split(int(convoluted.size(-1) / 2), -1)
        out = out * torch.sigmoid(gate)
        return out

class GLU_alt(nn.Module):
    def __init__(self,d_model,num_layers,patch_size=3,padding=1):#Dauphin's m_input= n_input= d_model
        super(GLU_alt,self).__init__()
        self.gated_convs = nn.ModuleList([GatedConvolution_alt(d_model,patch_size,padding) for _ in range(num_layers)])
    
    def forward(self,x):
        for convolution in self.gated_convs:
            x = convolution(x)
        return x


######################################################################

class Branched_conv_2d(nn.Module):
    def __init__(self, d_model, activation, out_d = 16,):
        super().__init__()
        self.convleft = nn.Conv2d(1, out_d, (1,1))
        self.convright = nn.Conv2d(1, out_d, (3,1), padding = (1, 0))
        self.activation = activation
        self.norm3 = nn.LayerNorm(out_d)

    def forward(self, x):
        x = x[:, None]
        x = self.activation(self.convleft(x)) + self.activation(self.convright(x))
        x.squeeze_(1)
        x = self.norm3(x)
        return x



class Branched_conv_stand(nn.Module):
    def __init__(self, d_model, activation, out_d = 256,):
        super().__init__()
        self.convleft = nn.Conv1d(in_channels=d_model, out_channels=out_d, kernel_size=1, bias=True)
        self.convright = nn.Conv1d(in_channels=d_model, out_channels=out_d, kernel_size=3, padding=1, bias=True)
        self.activation = activation
        self.norm3 = nn.LayerNorm(out_d)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.activation(self.convleft(x)) + self.activation(self.convright(x))
        x = x.transpose(1,2)
        x = self.norm3(x)
        return x

####################################################


class Group_End_2d_Wrap(nn.Module):
  def __init__(self, d_model):
    super().__init__()

    self.conv_group = nn.Conv2d(d_model, d_model, (9,1), groups = d_model, padding = (4,0))
    self.conv_end = nn.Conv2d(d_model, 1, (1,1))

  def forward(self, x):

    x = x[:, None]
    x = self.conv_group(x)
    x = self.conv_end(x)
    x.squeeze_(1)

    return x


class Group_End_1d_Wrap(nn.Module):
  def __init__(self, d_model):
    super().__init__()

    self.conv_group = nn.Conv1d(d_model, d_model, 9, groups = d_model, padding = 4, bias=True)
    self.conv_end = nn.Conv1d(d_model, 1, 1, bias=True)

  def forward(self, x):

    x = x.transpose(1,2)
    x = self.conv_group(x)
    x = self.conv_end(x)
    x = x.transpose(1,2)

    return x