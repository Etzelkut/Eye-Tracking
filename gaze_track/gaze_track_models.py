#if __name__ == '__main__' and __package__ is None:

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from basem.basic_dependency import *
from basem.blocks import Resize_Module, Pos_Emb, PositionwiseFeedForward
from basem.modules import Mish, Modified_Encoder_Layer, Encoder_Block, HeatMapExctract
from basem.functional import mish_f

from einops import rearrange

from gaze_track.utils import softargmax2d


# https://gist.github.com/mkocabas/4f56932afd21ce75e6b2e7d0c70488b8
class SpatialSoftmax(torch.nn.Module):
	def __init__(self, height, width, channel, temperature=None, data_format='NCHW', unnorm=False):
		super(SpatialSoftmax, self).__init__()
		self.data_format = data_format
		self.height = height
		self.width = width
		self.channel = channel
		self.unnorm = unnorm

		if temperature:
			self.temperature = Parameter(torch.ones(1) * temperature)
		else:
			self.temperature = 1.

		pos_x, pos_y = np.meshgrid(
			np.linspace(-1., 1., self.width),
			np.linspace(-1., 1., self.height)
		)
		pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
		pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
		self.register_buffer('pos_x', pos_x)
		self.register_buffer('pos_y', pos_y)

	def forward(self, feature):
		# Output:
		#   (N, C*2) x_0 y_0 ...
		if self.data_format == 'NHWC':
			feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
		else:
			feature = feature.view(-1, self.height * self.width)

		softmax_attention = F.softmax(feature / self.temperature, dim=-1)
		expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
		expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)

		if self.unnorm:
			w = float(self.width) - 1
			h = float(self.height) - 1
			expected_x = (expected_x * w + w) / 2.
			expected_y = (expected_y * h + h) / 2.

		expected_xy = torch.cat([expected_x, expected_y], 1)
		feature_keypoints = expected_xy.view(-1, self.channel, 2)

		return feature_keypoints

#



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

        new_size = one_patch_dim

        if feature_extract_hparams["resize_"]["add_resize"]:
            rm = feature_extract_hparams["resize_"]["resize_module"]
            self.patch_embedding = Resize_Module(type_module=rm["type_module"], 
                                size=one_patch_dim, new_size=rm["new_size"])
            new_size = rm["new_size"]
        else:
            print("no resize")
            self.patch_embedding = nn.Identity()

        self.dropout = nn.Dropout(feature_extract_hparams["dropout"])

        self.number_of_learn_params = feature_extract_hparams["number_of_learn_params"]
        n_of_learn_params = self.number_of_learn_params

        # goes in the end but wrote here for easier writing
        self.get_landmarks = lambda x: rearrange(x, 'b (h w) (p pd c) -> b c (h p) (w pd)', h = int(self.im_size[0]/self.patch_size),  p = self.patch_size, pd = self.patch_size, )
        if "alternative_landmarks" in feature_extract_hparams:
            if feature_extract_hparams["alternative_landmarks"]:
                print("adding alternative_landmarks in transformer")
                n_of_learn_params += 2
                self.get_landmarks = lambda x: x[:, 0:2]

    
        self.zero_class_token = nn.Parameter(torch.randn(1, n_of_learn_params, feature_extract_hparams["d_model_emb"]))

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
        self.image_extcractor = nn.Identity()
        
        self.add_train_land = nn.Identity()
        
        if feature_extract_hparams["add_additional_train_landmarks"]:
            if new_size == one_patch_dim:
                self.add_train_land = PositionwiseFeedForward(new_size, d_hid=new_size, activation=mish_f, glu=True)
            else:
                self.add_train_land = nn.Sequential(PositionwiseFeedForward(new_size, d_hid=new_size, activation=mish_f, glu=True),
                                            Resize_Module(type_module="fc", size=new_size, new_size=one_patch_dim)
                                            )
        elif new_size != one_patch_dim:
            self.add_train_land = Resize_Module(type_module="fc", size=new_size, new_size=one_patch_dim)


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
        
        x = self.image_extcractor(x[:, self.number_of_learn_params:])

        x = self.add_train_land(x)

        # c = self.channels, p = self.patch_size, h = self.im_size_h, w = self.im_size
        x = self.get_landmarks(x)   

        return x, feature_vector


class Gaze_Predictor(nn.Module):
    
    def __init__(self, params):
        super().__init__()
        self.params = params

        if self.params["feature_extractor_hparams"]["grayscale"]:
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


        h, w = params["feature_extractor_hparams"]["im_size"]
        c = 17

        alt_exist = False
        if "alternative_landmarks" in params["feature_extractor_hparams"]:
            print("alternative_landmarks is: ", params["feature_extractor_hparams"]["alternative_landmarks"])
            if params["feature_extractor_hparams"]["alternative_landmarks"]:
                alt_exist = True
                print("doint alt landmarks")
                self.heatmap_ex = lambda x: torch.flatten(x, start_dim=1)
                self.landmarks_extract = nn.Sequential(
                                       nn.Linear(d_model_emb * 2, d_model_emb * 2),
                                       Mish(),
                                       nn.Dropout(0.05),
                                       nn.Linear(d_model_emb * 2, c * 2),
                                       nn.Unflatten(1, (c, 2))
                                      )

        if "halfing" in params["feature_extractor_hparams"]:

            if alt_exist and params["feature_extractor_hparams"]["halfing"]:
                raise Exception("Both halfing and alternative_landmarks exist")

            print("hafing is: ", params["feature_extractor_hparams"]["halfing"])

            if params["feature_extractor_hparams"]["halfing"]:
                print("adding halfing for sure")
                add_pool_end = False
                if params["feature_extractor_hparams"]["halfing"]:
                    h, w = int(h * 0.5), int(w * 0.5)
                if "add_pool_end" in params["feature_extractor_hparams"]:
                    if params["feature_extractor_hparams"]["add_pool_end"]:
                        print("adding pool at the end")
                        add_pool_end = True
                
                self.heatmap_ex = HeatMapExctract(self.channels, params["feature_extractor_hparams"]["halfing"], add_pool_end)
        
        elif not alt_exist:
            self.heatmap_ex = HeatMapExctract(self.channels)
        
        self.landmarks_extract = SpatialSoftmax(h, w, c, temperature=1., unnorm=True)

    def forward(self, x):
        x, feature_vector = self.feature_extcractor(x)
        feature_vector = torch.flatten(feature_vector, start_dim=1)

        gaze = self.gaze_mlp(feature_vector)
        heatmaps = self.heatmap_ex(x)
        landmarks_out = self.landmarks_extract(heatmaps) #softargmax2d

        return gaze, heatmaps, landmarks_out
