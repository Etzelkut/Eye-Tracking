from basic_dependency import *
from subblocks import Swish, Mish
from functional import swish_f, mish_f, clones
from blocks import Attention_Block, Resize_Module, Pos_Emb, PositionwiseFeedForward, PositionwiseFeedForward_conv, Conv2Block


def activation_choose(types):
    if types == "mish":
        return mish_f
    elif types == "swish"
        return swish_f
    else:
        return torch.nn.functional.gelu



class Encoder_Layer(nn.Module):
    def __init__(self, layer_hparams):
        super().__init__()
        self.att = Attention_Block(layer_hparams["attention_block"])
        self.layer_norm = nn.LayerNorm(layer_hparams["d"])
        
        ff_activation_type = layer_hparams["ff_activation_type"]
        ff_glu = layer_hparams["ff_glu"]
        ff_dropout = layer_hparams["ff_dropout"]

        if layer_hparams["ff_type"] == "conv":
            self.ff = PositionwiseFeedForward_conv(layer_hparams["d"], activation = activation_choose(ff_activation_type), glu = ff_glu, dropout = ff_dropout)
        elif layer_hparams["ff_type"] == "fc":
            self.ff = PositionwiseFeedForward(layer_hparams["d"], activation = activation_choose(ff_activation_type), glu = ff_glu, dropout = ff_dropout)
    
    def forward(self, x):
        z = self.att(x)
        z = self.layer_norm(z + x) #residual
        z = self.ff(z)
        return z



#add just several version of glu like else

class GLU ...

class Modified_Encoder_Layer(nn.Module):
    def __init__(self, m_layer_hparams):
        super().__init__()



class Encoder_Block(nn.Module):
    
    def __init__(self, encoder_hparams):
        super().__init__()
        layer = Encoder_Layer(encoder_hparams)
        self.layers = clones(layer, encoder_hparams["layers_number"])        
        self.norm_after = nn.LayerNorm(encoder_hparams["d"]) if encoder_hparams["norm_after_block"] else nn.Identity()
        if encoder_hparams["alternative_weight_init"]:
            self.weight_init()
    
    def weight_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm_after(x)
        return x


class HeatMapExctract(nn.Module):
  def __init__(self, in_channels):
    super(HeatMapExctract, self).__init__()
    self.block1 = Conv2Block(in_channels, 6)
    self.block2 = Conv2Block(6, 12)
    self.block3 = Conv2Block(12, 24)
    self.block4 = Conv2Block(24, 34)

    self.downsample = nn.AvgPool2d(2, 2)

  def forward(self, x):
      x = self.block1(x)
      x = self.block2(x)
      x = self.block3(x)
      x = self.block4(x)

      x = self.downsample(x)

      return x