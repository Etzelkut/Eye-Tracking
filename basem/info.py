base_args = {
    "d": 40,
    "new_d": 128,
}


example_attention_block = {
    "type_module" : "gcn", #att, gcn, gcn_no_wn, linear, rpr, local_sa, light, dynamic
    "d": base_args["new_d"],
    "dropout": 0.05, # in some dropout do not exist
    "num_heads": 2, #if type_module == local_sa, then n_local_attn_heads + global = num_heads
    #params for local_sa
    "n_local_attn_heads": 0,
    "local_attn_window_size": 50,
    #params for glu
    "k_kernel_glu": 4,
    "downbot_glu": 2,
    #for rpr type
    "max_relative_positions_rpr": 8,
    #for dynamic and light
    #"kernel_size_dynamic": 30, 
}


example_encoder_hparams = {
    "attention_block": example_attention_block,
    "d": base_args["new_d"],
    "ff_type": "fc", #fc, conv
    "ff_activation_type": "mish", #swish or gelu
    "ff_glu": False,
    "ff_dropout": 0.05,
    # encoder block params
    "layers_number":2,
    "norm_after_block":False,
    "alternative_weight_init": False,
}




example_emb = {
    "add_emb": True, 
    "emb_module": {
        "type_module": "axial",
        "d": base_args["new_d"], # dimensions
        "dropout": 0.1,
    }
}


example_resize = {
    "add_resize": True, 
    "resize_module": {
        "type_module": "1dconv",
        "size": base_args["d"], # dimensions
        "new_size": base_args["new_d"],
    },
}
