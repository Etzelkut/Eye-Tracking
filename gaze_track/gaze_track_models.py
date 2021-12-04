#if __name__ == '__main__' and __package__ is None:

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from basem.basic_dependency import *
from basem.blocks import Resize_Module, Pos_Emb
from basem.modules import Mish, Modified_Encoder_Layer, Encoder_Block, HeatMapExctract

from einops import rearrange

from gaze_track.utils import softargmax2d

class ViT_pos_emb(nn.Module):
    
    def __init__(self, n_patch, d_model_emb, number_of_learn_params):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, n_patch + number_of_learn_params, d_model_emb))
        print("running classic ViT_pos_emb")
    def forward(self, x):
        x = x + self.positional_embedding
        return x


class Transf_Feature_Extract(nn.Module):
    
    def __init__(self, feature_extract_hparams):
        super().__init__()
        
        if feature_extract_hparams["grayscale"]:
            channels = 1
        else:
            channels = 3
        
        self.im_size, self.patch_size, d_model_emb = feature_extract_hparams["im_size"], feature_extract_hparams["patch_size"], feature_extract_hparams["d_model_emb"]
        n_patch = int(self.im_size[1]/self.patch_size) * int(self.im_size[0]/self.patch_size)
        
        one_patch_dim = int(channels * (self.patch_size)**2)

        if feature_extract_hparams["resize_"]["add_resize"]:
            rm = eature_extract_hparams["resize_"]["resize_module"]
            self.patch_embedding = Resize_Module(type_module=rm["type_module"], 
                                size=one_patch_dim, new_size=rm["new_size"])
        else:
            print("no resize")
            self.patch_embedding = nn.Identity()

        self.dropout = nn.Dropout(feature_extract_hparams["dropout"])

        self.number_of_learn_params = model_dict["number_of_learn_params"]
        self.zero_class_token = nn.Parameter(torch.randn(1, self.number_of_learn_params, feature_extract_hparams["d_model_emb"]))

        if feature_extract_hparams["pos_emb"]["add_emb"]:
            pe = feature_extract_hparams["pos_emb"]["emb_module"]
            self.positional_embedding = Pos_Emb(type_module=pe["type_module"], 
                                    d=pe["d"], dropout=pe["dropout"])
        else:
            self.positional_embedding = ViT_pos_emb(n_patch, d_model_emb, self.number_of_learn_params)

        if feature_extract_hparams["encoder_type"] == "evolved":
            self.encoder = Modified_Encoder_Layer(feature_extract_hparams["encoder_params"])
        elif feature_extract_hparams["encoder_type"] == "transformer":
            self.encoder = Encoder_Block(feature_extract_hparams["encoder_params"])

        self.feature_extcractor = nn.Identity()

    def forward(self, x):
        # 1 1 (96) (160) - >  1 (60: 6 * 10) (256: 16 * 16)
        x = rearrange(x, 'b c (h p) (w pd) -> b (h w) (p pd c)', p = self.patch_size, pd = self.patch_size)
        x = self.patch_embedding(x)
        x = self.dropout(x)

        b, n, d = x.shape

        zero_class_token = torch.repeat_interleave(self.zero_class_token, repeats = b, dim=0)
        x = torch.cat((zero_class_token, x), dim=1)

        x = self.positional_embedding(x)
        x = self.encoder(x)

        feature_vector = self.feature_extcractor(x[:, 0:self.number_of_learn_params])

        # c = self.channels, p = self.patch_size, h = self.im_size_h, w = self.im_size
        x = rearrange(x[:, self.number_of_learn_params:], 'b (h w) (p pd c) -> b c (h p) (w pd)', h = int(self.im_size[0]/self.patch_size),  p = self.patch_size, pd = self.patch_size, )   

        return x, feature_vector


class Gaze_Predictor(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        self.params = params

        if self.params["grayscale"]:
            self.channels = 1
        else:
            self.channels = 3

        self.feature_extcractor = Transf_Feature_Extract(params["feature_extractor_hparams"])

        d_model_emb = params["feature_extractor_hparams"]["d_model_emb"] * params["feature_extractor_hparams"]["number_of_learn_params"]

        self.gaze_mlp = nn.Sequential(
                                       nn.Linear(d_model_emb, d_model_emb),
                                       Mish(),
                                       nn.Dropout(self.params["mlp_drop"]),
                                       nn.Linear(d_model_emb, self.params["gaze_size"]),
                                      )

        self.heatmap_ex = HeatMapExctract(self.channels)

    def forward(self, x):
        x, feature_vector = self.feature_extcractor(x)
        feature_vector = torch.flatten(feature_vector, start_dim=1)

        gaze = self.gaze_mlp(feature_vector)
        heatmaps = self.heatmap_ex(x)
        landmarks_out = softargmax2d(heatmaps)

        return gaze, heatmaps, landmarks_out
