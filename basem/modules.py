from basic_dependency import *
from subblocks import Swish, Mish
from functional import swish_f, mish_f, clones
from blocks import Attention_Block, Resize_Module, Pos_Emb, PositionwiseFeedForward, PositionwiseFeedForward_conv, Conv2Block



class Updated_old_Re_model(nn.Module):
    def __init__(self, re_dict):
        super(Updated_old_Re_model, self).__init__()
        re_dict_ = copy.deepcopy(re_dict)
        model_dict = copy.deepcopy(re_dict["model_dict"])
        
        self.resize_d = False
        if model_dict["add_resize_for_d"]:
            self.resize_d = True
            typo = model_dict["type_resize_d"]
            self.embeding = resize_module(typo, model_dict["d"], model_dict["resize_d"])
            model_dict["d"] =  model_dict["resize_d"]
            re_dict_["main_block"]["d"] = model_dict["resize_d"]
            re_dict_["add_block"]["d"] = model_dict["resize_d"]

        
        self.number_of_learn_params = model_dict["number_of_learn_params"]
        self.zero_class_token = nn.Parameter(torch.randn(1, self.number_of_learn_params, model_dict["d"]))

        self.pos_emb = Pos_Emb(model_dict["d"], model_dict["dropout_pos_emb"], model_dict["position_emb"])
        
        #start
        self.norm = nn.LayerNorm(model_dict["d"])

        self.main_block = nn.ModuleList([Main_Block(re_dict_["main_block"])])
        if model_dict["number_of_main"] > 1:
            self.main_block.extend([Main_Block(re_dict_["main_block"]) for i in range(1, model_dict["number_of_main"])])


        self.norm2 = nn.LayerNorm(model_dict["d"])
        self.convleft = nn.Conv2d(1, 8, (1,1))
        self.convright = nn.Conv2d(1, 8, (3,1), padding = (1, 0))

        self.norm3 = nn.LayerNorm(model_dict["d"])
        self.conv_group = nn.Conv2d(8,8, (9,1), groups = 8, padding = (4,0))
        self.conv_end = nn.Conv2d(8,1, (1,1))

        self.norm4  = nn.LayerNorm(model_dict["d"])

        self.add_block = False
        if model_dict["add_block"]:
            self.add_block = True
            self.additional_block = nn.ModuleList([Main_Block(re_dict_["add_block"])])
            if model_dict["number_of_add"] > 1:
                self.additional_block.extend([Main_Block(re_dict_["add_block"]) for i in range(1, model_dict["number_of_add"])])

        #end

        self.feature_extcractor = nn.Identity()
        d = model_dict["d"]

        self.sum_end = False

        indx = self.number_of_learn_params
        if model_dict["sum_end"]:
          indx += 1
          print("sum_end")
          self.sum_end = True
          in_feat = int(d * indx)
          out_feat = int(d * indx * 1.5)
        else:
          in_feat = d * indx
          out_feat = int(d * indx * 1.5)

        self.w1 = nn.Linear(in_features = in_feat, out_features = out_feat)
        self.activation = Mish()
        self.dropout = nn.Dropout(model_dict["classificator_dropout"])
        self.w2 = nn.Linear(in_features = out_feat, out_features = model_dict["num_classes"])

        self.enable_softmax = False
        if model_dict["enable_softmax"]:
          self.enable_softmax = True

        self.type_param_insert = model_dict["type_param_insert"]
       




        #start


        self.convleft = nn.Conv2d(1, 8, (1,1))
        self.convright = nn.Conv2d(1, 8, (3,1), padding = (1, 0))

        self.norm3 = nn.LayerNorm(model_dict["d"])
        self.conv_group = nn.Conv2d(8,8, (9,1), groups = 8, padding = (4,0))
        self.conv_end = nn.Conv2d(8,1, (1,1))

        self.norm4  = nn.LayerNorm(model_dict["d"])

        self.add_block = False
        if model_dict["add_block"]:
            self.add_block = True
            self.additional_block = nn.ModuleList([Main_Block(re_dict_["add_block"])])
            if model_dict["number_of_add"] > 1:
                self.additional_block.extend([Main_Block(re_dict_["add_block"]) for i in range(1, model_dict["number_of_add"])])

        #end

class Modified_Encoder_Layer(nn.Module):
    def __init__(self, m_layer_hparams):
        super().__init__()
        modules_list = []
        modules_list.append(nn.LayerNorm(m_layer_hparams["d"]))
        
        for i in range(m_layer_hparams["number_of_main"]):
            modules_list.append(Attention_Block(m_layer_hparams["main_attention_block"]))

        modules_list.append(nn.LayerNorm(m_layer_hparams["d"]))






        modules_list.append(nn.LayerNorm(m_layer_hparams["d"]))
        modules_list.append(nn.LayerNorm(m_layer_hparams["d"]))
        modules_list.append(nn.LayerNorm(m_layer_hparams["d"]))



    def forward(self, x):
        return x




def activation_choose(types):
    if types == "mish":
        return mish_f
    elif types == "swish":
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