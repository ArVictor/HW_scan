import datetime
import time
import os
import numpy as np


def get_model_name(local_dict):
    file_name = local_dict["output_path"]
    list_names = ["solver_name", "poisson_method"]
    for k, v in local_dict.items():
        if type(v) == int or type(v) == float or k in list_names:
            file_name = file_name + "_" + k + str(v)

    file_name = file_name.replace("_unrolled_steps", "u")
    file_name = file_name.replace("_decay_epochs", "dec")
    file_name = file_name.replace("_epochs_steps", "epo")
    file_name = file_name.replace("_save_model_skips", "sms")
    file_name = file_name.replace("_valid_skips", "vs")
    file_name = file_name.replace("_valid_epochs", "ve")
    file_name = file_name.replace("_test_epochs", "te")
    file_name = file_name.replace("_kappa_coeff", "kap")
    file_name = file_name.replace("_arakawa_coeff", "ara")
    file_name = file_name.replace("_solver_name", "sol")
    file_name = file_name.replace("_poisson_method", "pois")
    file_name = file_name.replace("_min_fps", "fps")
    file_name = file_name.replace("_batch_size_start", "bs")
    file_name = file_name.replace("_batch_size_step", "bt")
    file_name = file_name.replace("_batch_size_end", "be")
    file_name = file_name.replace("_buffer_size", "buf")
    file_name = file_name.replace("_weight_average", "WA")
    file_name = file_name.replace("_valid_batch", "vb")
    file_name = file_name.replace("_steps_simulation", "sts")
    file_name = file_name.replace("_time_window_to_keep", "twtk")
    file_name = file_name.replace("_speed", "s")
    file_name = file_name.replace("buffer", "bf")
    file_name = file_name.replace("_layers", "l")
    file_name = file_name.replace("_filters", "f")
    file_name = file_name.replace("_n_blocks_invariants", "nb")
    file_name = file_name.replace("_early_stopping_patience", "esp")
    file_name = file_name.replace("_optimizer_str", "o")
    file_name = file_name.replace("_temporal_causality_eps", "tc")
    file_name = file_name.replace("_model_type", "m")
    file_name = file_name.replace("resnet", "rn")
    file_name = file_name.replace("unet", "un")
    file_name = file_name.replace("_valid_warmup", "vw")
    file_name = file_name.replace("_SAM_warmup", "sw")
    

    str_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    while os.path.exists(file_name + "_" + str_time):
        time.sleep(2 + np.random.rand() * 5)
        str_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = file_name + "_" + str_time

    return file_name
