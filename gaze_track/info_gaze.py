from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from basem.info import example_evolved_encoder_hparams


base_args = {
    "d": 0,
    "new_d": 256,
    "im_size": (96, 160),
    "patch_size": 16,
    "grayscale": True,
    "max_epochs": 100, 
    "halfing": False,

    "alternative_landmarks": False,
    "add_heatmap_loss": False,
}

example_emb = {
    "add_emb": False, # if false will run normal embeding from the ViT
    "emb_module": {
        "type_module": "axial", # const
        "d": base_args["new_d"], # dimensions
        "dropout": 0.1,
    }
}


example_resize = {
    "add_resize": True, 
    "resize_module": {
        "type_module": "fc", #fc, 1dconv
        "size": base_args["d"], # dimensions
        "new_size": base_args["new_d"],
    },
}


dataset_hparams = {
    "img_dir": None,
    "grayscale": base_args["grayscale"],
    "im_size": base_args["im_size"],
    "batch_size": 32,
    "num_workers":2,
    "dataloader_shuffle": True,

    "halfing": base_args["halfing"],

    "alternative_landmarks": base_args["alternative_landmarks"],
}


example_feature_extract_hparams = {
    "grayscale": base_args["grayscale"],
    "im_size": base_args["im_size"],
    "patch_size": base_args["patch_size"],
    "d_model_emb": base_args["new_d"],
    "resize_": example_resize,
    "pos_emb": example_emb,
    "dropout": 0.05,
    "number_of_learn_params": 1,
    "encoder_type": "evolved", # transformer
    "encoder_params": example_evolved_encoder_hparams,

    "alternative_landmarks": base_args["alternative_landmarks"],
    
    "add_additional_train_landmarks": True,

    "halfing": base_args["halfing"],
    "add_pool_end": False,
}

train_hparams_example = {
    "optimizer": "adamW", # "belief", "ranger_belief", "adam", adamW
    "lr": 3e-4, #
    "epochs": base_args['max_epochs'], #
    #
    "add_sch": False,
    #
    #belief
    "eplison_belief": 1e-16,
    "beta": [0.9, 0.999], # not used
    "weight_decouple": True, 
    "weight_decay": 1e-4,
    "rectify": True,

}

example_model_hparams = {
    "type": "trans_based",
    "feature_extractor_hparams": example_feature_extract_hparams,

    "alternative_landmarks": base_args["alternative_landmarks"],
    "add_heatmap_loss": base_args["add_heatmap_loss"],
    "im_size": base_args["im_size"],

    "mlp_drop": 0.05,
    "gaze_size": 2,
    "traning": train_hparams_example,
}