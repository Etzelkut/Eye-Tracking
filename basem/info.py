base_args = {
    "d": 0,
    "new_d": 256,
}


example_attention_block = {
    "type_module" : "gcn", #att, gcn, gcn_no_wn, linear, rpr, local_sa, glu_alt, light, dynamic
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
    #for glu alt
    "alt_num_layers": 1,
    "alt_patch_size": 3,
    "alt_padding": 1,
}


example_branch = {
    "type_module": "1d", #2d
    "d_model": base_args["new_d"],
    "out_d": 256,
}


example_evolved_encoder_hparams = {
    "d": base_args["new_d"],
    "number_of_main": 1,
    "main_attention_block": example_attention_block,
    "branched_conv": example_branch,
    
    "number_of_add": 1,
    "add_attention_block": example_attention_block,

    "norm_after_block":False,
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
