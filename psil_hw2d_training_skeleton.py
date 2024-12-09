import tensorflow as tf

import fire
import numpy as np
from phiml import math
from phiml import nn
from typing import Iterable, Union
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import os
import copy
import time


# Local imports
import tf_hw2d
import hw2d
from hw2d.utils.namespaces import Namespace
from tf_hw2d.model import HW
from hw2d.model import HW as numba_HW
from tf_hw2d.utils.tf_io import (
    get_save_params,
    create_appendable_h5,
    save_to_buffered_h5,
    append_h5,
    continue_h5_file,
)

from tf_hw2d.utils.plot.movie import create_movie

from tf_hw2d.utils.tf_run_properties import calculate_properties

from tf_hw2d.utils.plot.timetrace import plot_timetraces

from tf_hw2d.neural_network import tf_loss, tf_corrector, tf_resnet, tf_resnetSCS, tf_unet, tf_tools, tf_update_weights

from tf_hw2d.utils import tf_npdict_to_tensordict, tf_model_naming

import tensorflow_addons as tfa

from tensorflow.python.compiler.tensorrt import trt_convert as trt

import subprocess as sp
import os

def get_gpu_memory():
    # command = "nvidia-smi"# -q –d CLOCK"
    # memory_free_info = sp.check_output(command.split()).decode('ascii')
    # print("CLOCK GPU", memory_free_info)
    # #command = "nvidia-smi --format=csv"
    # #memory_free_info = sp.check_output(command.split()).decode('ascii')
    # print("NVIDIA-SMI GPU", memory_free_info)
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

print("free GPU mem start:", get_gpu_memory())
print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])

tf.get_logger().setLevel('ERROR')

def training(
    train_dataset_path: Iterable[str] = [""],
    valid_dataset_path: Iterable[str] = [""],
    test_dataset_path: Iterable[str] = [""],
    unrolled_steps: int = 1,
    lr: float = 1e-4,
    optimizer_str: str = "adam",
    rho: float = 0.0,
    SAM_warmup: int = 5,
    temporal_causality_eps: float = 0.0,
    decay_epochs: int = 0,
    batch_size_start: int = 1,
    batch_size_step: int = 3,
    batch_size_end: int = 10,
    epochs_steps: int = 1000,
    save_model_skips: int = 1000,
    valid_skips: int = 20,
    valid_epochs: int = 10,
    valid_batch: int = 10,
    valid_warmup: int = 50,
    steps_simulation: int = 50,
    tensorrt_inference: bool = True,
    model_type: str = "resnet",
    layers: int = 5,
    filters: int = 16,
    weight_average: int = 0,
    batch_norm: bool = False,
    early_stopping_patience: int = 0,
    load_model: str = "",
    solver_name: str = "rk4",
    N: int = 3,
    nu: float = 5.0e-08,
    c1: float = 1.0,
    kappa_coeff: float = 1.0,
    arakawa_coeff: float = 1.0,
    poisson_method: str = "fourier",
    time_window_to_keep: int = 700,
    seed: int or None = None,
    snaps: int = 1,
    buffer_size: int = 100,
    output_path: str = "",
    continue_file: bool = False,
    movie: bool = True,
    min_fps: int = 10,
    dpi: int = 75,
    speed: int = 5,
    debug: bool = False,
    properties: Iterable[str] = [
        "gamma_n",
        "gamma_n_spectral",
        #"gamma_c",
        "energy",
        "thermal_energy",
        "kinetic_energy",
        "enstrophy",
        "enstrophy_phi",
    ],
    plot_properties: Iterable[str] = (
        "enstrophy",
        "energy",
        "kinetic_energy",
        "thermal_energy",
    ),
    evaluation_properties: Iterable[str] = (
        "gamma_n_spectral",
        "enstrophy",
    ),
    n_blocks_invariants: int = 4,
    keep_all_simulation_data: bool = False,
):
    """
    Trains the model with the given parameters.

    Args:
        train_dataset_path (Iterable[str]): Path to the training data or multiple datasets as list of names.
        valid_dataset_path (Iterable[str]): Path to the validation data or multiple datasets as list of names.
        test_dataset_path (Iterable[str]): Path to the test data or multiple datasets as list of names.
        unrolled_steps (int, optional): Number of steps unrolled for neural network training. Defaults to 1.
        lr (float, optional): Learning rate. Defaults to 0.0001.
        optimizer_str (str, optional): Optimizer, can be 'adam' or 'SGD'. Defaults to 'adam'.
        rho (float, optional): Rho parameter for gradient ascent in Sharpness-Aware Minimization. Defaults to 0.0.
        SAM_warmup (int, optional): Number of iterations before using Sharpness-Aware Minimization. Defaults to 5.
        temporal_causality_eps (float, optional): Epsilon parameter for temporal causality loss. Defaults to 0.0. if <=0, temporal causality is not used. See https://arxiv.org/pdf/2203.07404.pdf RESPECTING CAUSALITY IS ALL YOU NEED FOR TRAINING PHYSICS-INFORMED NEURAL NETWORKS.
        decay_epochs (int, optional): If non-zero, the number of epochs over which the learning rate is decayed (CosineDecayRestarts). Defaults to 0.
        batch_size_start (int, optional): Size of initial training batch. Defaults to 1.
        batch_size_step (int, optional): Size of steps for training batch. Defaults to 3.
        batch_size_end (int, optional): Size of final training batch. Defaults to 10.
        epochs_steps (int, optional): Number of training epochs_steps. Defaults to 1000.
        save_model_skips (int, optional): Number of training epochs_steps to skip between model saves. Defaults to 1000.
        valid_skips (int, optional): Number of training epochs_steps to skip between valid/test evaluations. Defaults to 20.
        valid_epochs (int, optional): Number of validation steps. Defaults to 10.
        valid_batch (int, optional): Number of validation batches. Defaults to 10.
        valid_warmup (int, optional): Number of epochs before starting validation evaluations. Defaults to 50.
        steps_simulation (int, optional): Number of testing steps with gamma_n distance evaluation. Defaults to 50.
        tensorrt_inference (bool, optional): Use a tensorrt-converted model for validation and test inferences. There is a significant overhead from tensorrt conversion, but the model is also significantly faster afterwards. Defaults to True.
        model_type (str, optional): Name of the model type, 'resnet', 'unet', or 'scs'.
        layers (int, optional): Number of hidden layers of the resnet. Defaults to 5.
        filters (int, optional): Width of each hidden layers of the resnet. Defaults to 16.
        weight_average (int, optional): Number of averaging steps from the last steps of training. Defaults to 0, no averaging.
        batch_norm (bool, optional): Use batch normalization. Default to False.
        early_stopping_patience (int, optional): If >1 the number of validation steps to wait for improvement before stopping the training.
        solver_name (int, optional): Integrate with Euler, rk2 or rk4. Defaults to rk4.
        N (int, optional): Dissipation exponent's half value. Defaults to 3.
        nu (float, optional): Viscosity. Suggested: 5e-10 for coarse-large, 1e-4 for fine-small. Defaults to 5.0e-08.
        c1 (float, optional): Transition scale between hydrodynamic and adiabatic. Suggested values: 0.1, 1, 5. Defaults to 1.0.
        kappa_coeff (float, optional): Coefficient of d/dy phi. Defaults to 1.0.
        arakawa_coeff (float, optional): Coefficient of Poisson bracket [A,B] implemented with Arakawa Scheme. Defaults to 1.0.
        poisson_method (str, optional): Method to solve the Poisson equation. Either "fourier" or "CG".
        time_window_to_keep (int): Length of the time window to train the models on, starting from the end. Defaults to 700.
        seed (int or None, optional): Seed for random number generation. Defaults to None.
        init_type (str, optional): Initialization method. Choices: 'fourier', 'sine', 'random', 'normal'. Defaults to 'normal'.
        init_scale (float, optional): Scaling factor for initialization. Defaults to 0.01.
        snaps (int, optional): Snapshot intervals for saving. Defaults to 1.
        buffer_size (int, optional): Size of buffer for storage. Defaults to 100.
        output_path (str, optional): Where to save the simulation data. Defaults to ''.
        continue_file (bool, optional): If True, continue with existing file. Defaults to False.
        movie (bool, optional): If True, generate a movie out of simulation. Defaults to True.
        min_fps (int, optional): Parameter for movie generation. Defaults to 10.
        dpi (int, optional): Parameter for movie generation. Defaults to 75.
        speed (int, optional): Parameter for movie generation. Defaults to 5.
        debug (bool, optional): Enable or disable debug mode. Defaults to False.
        properties (Iterable[str], optional): List of properties to calculate for the saved file.
        plot_properties (Iterable[str], optional): List of properties to plot a timetrace for.
        evaluation_properties (Iterable[str], optional): List of properties to to use for model evaluation.
        n_blocks_invariants (int): Number of blocks to compute standard deviation of block's mean for evaluation.
        keep_all_simulation_data (bool, optional): Enable or disable removing of simulation data after evaluation metrics have been calculated. Defaults to False.

    Returns:
        None: The function saves simulation data or generates a movie as specified.
    """
    print("locals:", locals())
    if output_path:
        output_path = tf_model_naming.get_model_name(locals())
    print("output_path:", output_path)
    # math.use('numpy')
    print("free GPU mem before use tensorflow:", get_gpu_memory())
    print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
    print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
    math.use("tensorflow")
    math.set_global_precision(32)  # does it change something??
    print("free GPU mem after use tensorflow:", get_gpu_memory())
    print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
    print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])

    # Initializations of Seeds
    math.seed(seed)
    dataset_names = ["train", "valid", "test"]
    # Define load path
    # Load path
    print("free GPU mem before load_h5py_data:", get_gpu_memory())
    print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
    print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
    parameters_list, np_data_dict_list = load_h5py_data(
        train_dataset_path, valid_dataset_path, test_dataset_path, dataset_names
    )
    print("free GPU mem after load_h5py_data:", get_gpu_memory())
    print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
    print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
    # Create Datasets
    input_x_names = ["phi"]
    output_y_names = ["density", "omega"]
    # Always make sure dx and k0 are INVARIANT for the train, valid, test
    for key_set in dataset_names:
        for p in parameters_list[key_set]:
            if parameters_list["train"][0]["dx"] - p["dx"] != 0:
                print("ERROR TRAINING SETS HAVE DIFFERENT dx.")
            if parameters_list["train"][0]["k0"] - p["k0"] != 0:
                print("ERROR TRAINING SETS HAVE DIFFERENT k0.")
    print("parameters_list:", parameters_list)
    # Physics
    # Always make sure dx and k0 are INVARIANT for the train, valid, test
    physics_params = dict(
        dx=parameters_list["train"][0]["dx"],
        N=N,
        #c1=c1,
        c1=math.tensor(c1["train"], math.batch('batch')),
        nu=nu,
        k0=parameters_list["train"][0]["k0"],
        arakawa_coeff=arakawa_coeff,
        kappa_coeff=kappa_coeff,
        poisson_method=poisson_method,
        TEST_CONSERVATION=False,
    )
    print("physics_params:", physics_params)
    hw = HW(**physics_params, debug=debug)

    # Full Predictions
    if solver_name == "euler":
        integration_step = hw.euler_step
    elif solver_name == "rk2":
        integration_step = hw.rk2_step
    elif solver_name == "rk4":
        integration_step = hw.rk4_step
    #integration_step = tf.recompute_grad(integration_step)

    # Set Neural Network Model
    print("free GPU mem before cnn:", get_gpu_memory())
    print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
    print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
    if load_model:
        cnn = tf.keras.models.load_model(load_model)
        print("load_model:", load_model)
    else:
        if model_type == "resnet":
            cnn = tf_resnet.build_resnet(
                in_channels=len(input_x_names),
                #in_channels=9,  #JUST FOR THE TEST WITH GRADIENTS AS INPUTS
                out_channels=len(output_y_names),
                layers=[filters for _ in range(layers)],
                kernel_size=5,
                batch_norm=batch_norm,
                activation_layer=tf.keras.layers.LeakyReLU,
                # activation_layer=tf.keras.layers.ELU,
            )
        elif model_type == "unet":
            cnn = tf_unet.build_Unet(
                input_shape=[parameters_list["train"][0]["grid_pts"],parameters_list["train"][0]["grid_pts"],1],
                output_channels=2,
                base_filters  = 16,
                depth         = 4,
                dropout       = 0.2,
                activation    = 'relu',
                use_batchnorm = True,
            )
        elif model_type == "scs":
            cnn = tf_resnetSCS.build_resnetSCS(
                in_channels=len(input_x_names),
                out_channels=len(output_y_names),
                layers=[filters for _ in range(layers)],
                kernel_size=3,
                batch_norm=batch_norm,
                activation_layer=tf.keras.layers.LeakyReLU,
                # activation_layer=tf.keras.layers.ELU,
                input_x_z=parameters_list["train"][0]["grid_pts"],
            )
    print("cnn", cnn)
    # cnn_recompute = tf.recompute_grad(cnn)
    cnn.summary()
    print("cnn.losses", cnn.losses)
    print("free GPU mem after cnn:", get_gpu_memory())
    print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
    print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])

    np_data_dict_indices_list = {
        key_set: [
            np.arange(np_data_dict["density"].shape[0], dtype=np.int64)
            for np_data_dict in np_data_dict_list[key_set]
        ]
        for key_set in dataset_names
    }
    print("Init np_data_dict_indices_list:", np_data_dict_indices_list)
    # ROBIN ACTUALLY DIDN'T USE THE FIRST PART OF THE SIMULATION! IT DIDN'T CONVERGE!!
    #This removes the first third of the simulations
    # for key_set in dataset_names:
    #     for i_index_list, np_data_dict in enumerate(np_data_dict_list[key_set]):
    #         np_data_dict_indices_list[key_set][
    #             i_index_list
    #         ] = np_data_dict_indices_list[key_set][i_index_list][
    #             int(0.3 * np_data_dict["density"].shape[0]) :
    #         ]
    #         for k in np_data_dict.keys():
    #             print("before 0.3", k, "np_data_dict[k].shape", np_data_dict[k].shape)
    #             np_data_dict[k] = np_data_dict[k][
    #                 int(0.3 * np_data_dict[k].shape[0]) :,
    #                 ...,
    #             ]
    #             print("0.3", k, "np_data_dict[k].shape", np_data_dict[k].shape)
    # print("0.3* np_data_dict_indices_list:", np_data_dict_indices_list)
    #This keeps the last 700/dt steps of the simulations
    for key_set in dataset_names:
        print("free GPU mem 700:", get_gpu_memory())
        print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
        print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
        for i_index_list, np_data_dict in enumerate(np_data_dict_list[key_set]):
            np_data_dict_indices_list[key_set][
                i_index_list
            ] = np_data_dict_indices_list[key_set][i_index_list][
                -int(time_window_to_keep / parameters_list[key_set][i_index_list]["dt"]) :
            ]
            for k in np_data_dict.keys():
                print("before -time_window_to_keep/dt", k, "np_data_dict[k].shape", np_data_dict[k].shape)
                np_data_dict[k] = np_data_dict[k][
                    -int(time_window_to_keep / parameters_list[key_set][i_index_list]["dt"]) :,
                    ...,
                ]
                print("-time_window_to_keep/dt", k, "np_data_dict[k].shape", np_data_dict[k].shape)
    print("-time_window_to_keep/dt np_data_dict_indices_list:", np_data_dict_indices_list)
    # For the old data, the last slices are just repeated. I made a mistake when copying the files.
    for key_set in dataset_names:
        print("free GPU mem -100:", get_gpu_memory())
        print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
        print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
        for i_index_list, np_data_dict in enumerate(np_data_dict_list[key_set]):
            np_data_dict_indices_list[key_set][
                i_index_list
            ] = np_data_dict_indices_list[key_set][i_index_list][:-100]
            for k in np_data_dict.keys():
                print("before -100", k, "np_data_dict[k].shape", np_data_dict[k].shape)
                np_data_dict[k] = np_data_dict[k][
                    :-100,
                    ...,
                ]
                print("-100", k, "np_data_dict[k].shape", np_data_dict[k].shape)
    print("-100* np_data_dict_indices_list:", np_data_dict_indices_list)
    #Make sure all sets have the same length as the first one!
    length_first_dataset = len(np_data_dict_indices_list["train"][0])
    for key_set in dataset_names:
        print("free GPU mem equalize:", get_gpu_memory())
        print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
        print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
        for i_index_list, np_data_dict in enumerate(np_data_dict_list[key_set]):
            print("CPU GPU mem equalize file:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
            np_data_dict_indices_list[key_set][
                i_index_list
            ] = np_data_dict_indices_list[key_set][i_index_list][:length_first_dataset]
            for k in np_data_dict.keys():
                print("CPU GPU mem equalize field:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
                print("before equalize length_first_dataset", k, "np_data_dict[k].shape", np_data_dict[k].shape)
                np_data_dict[k] = np_data_dict[k][
                    :length_first_dataset,
                    ...,
                ]
                print("length_first_dataset", k, "np_data_dict[k].shape", np_data_dict[k].shape)
    print("equalize lengths:length_first_dataset np_data_dict_indices_list:", np_data_dict_indices_list)

    # Correction
    # ROBIN COULD DO IT DIFFERENTLY: INSTEAD OF STD FOR THE ENTIRE DATA SET, COMPUTE STD PER SLICE AND TAKE MEAN! BUT ONLY A FEW PERCENT DIFFERENCE
    # Data statistics are only from the training set!
    (
        mean_omega,
        # std_omega,
        std_omega_robin,
        mean_density,
        # std_density,
        std_density_robin,
        mean_phi,
        std_phi,
        # ) = get_data_mean_std_robin(np_data_dict_list["train"]) #No he actually does the correct mean(std) ?
    ) = get_data_mean_std_slices(np_data_dict_list["train"])
    (_, std_omega, _, std_density, _, _) = get_data_mean_std_slices_diff(
        np_data_dict_list["train"]
    )  # We only overwrite std_omega and density as those are the corrections we apply to the fields
    print(
        "mean_omega, std_omega, mean_density, std_density, mean_phi, std_phi",
        mean_omega,
        std_omega,
        mean_density,
        std_density,
        mean_phi,
        std_phi,
    )

    # parameters_list[0] only used to get "dx" so any input dataset should be correct
    corrector_step = tf_corrector.get_corrector_function(
        mean_phi,
        std_omega,
        std_density,
        std_phi,
        hw,
        parameters_list["train"][
            0
        ],  # parameters_list["train"] only used for "dx" so it's ok to pass the "train" parameters explicitely
    )

    losses_list = {key_set: [] for key_set in dataset_names}
    is_first_progressive_unrolled_steps = True
    # for progressive_unrolled_steps in range(3, unrolled_steps + 1, 15):
    list_unrolled_steps = [3]
    if unrolled_steps != list_unrolled_steps[-1]:
        list_unrolled_steps.append(unrolled_steps)
    batch_size_list = list(range(batch_size_start, batch_size_end+1, batch_size_step))

    # for progressive_unrolled_steps in list_unrolled_steps:
    progressive_unrolled_steps = list_unrolled_steps[-1]

    # If we loop over unrolled_steps, this needs to be in the loop, and the decimation has to be corrected! 
    np_data_dict_shuffled_unrolled_list = np_data_dict_list
    np_data_dict_indices_shuffled_list = np_data_dict_indices_list
    print(
        "deep copy np_data_dict_indices_shuffled_list:",
        np_data_dict_indices_shuffled_list,
    )
    sort_and_extract_windows_vectorized(
        dataset_names,
        np_data_dict_shuffled_unrolled_list,
        is_first_progressive_unrolled_steps,
        np_data_dict_indices_shuffled_list,
        progressive_unrolled_steps,
    )

    #Create directory for the saved models
    if not os.path.exists( output_path + "_models"):
        os.makedirs( output_path + "_models")

    # If the optimizer is defined outside, does it keep the running mean of the gradient between the different progressive_unrolled_steps?
    # We need a full reset, maybe defining it inside the loop achieves this
    # optimizer = nn.adam(cnn, learning_rate=lr)
    if decay_epochs < 2:
        if optimizer_str == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=0.001)
        elif optimizer_str == 'adamW':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, clipnorm=0.001)
        elif optimizer_str == 'lion':
            optimizer = tf.keras.optimizers.Lion(learning_rate=lr, clipnorm=0.001)
        elif optimizer_str == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, clipnorm=0.001)
        # optimizer = tfa.optimizers.LAMB(learning_rate=lr)
    else:
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr,
            first_decay_steps=decay_epochs,
        )
        if optimizer_str == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_decayed_fn, clipnorm=0.001
            )
        elif optimizer_str == 'adamW':
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_decayed_fn, clipnorm=0.001
            )
        elif optimizer_str == 'lion':
            optimizer = tf.keras.optimizers.Lion(
                learning_rate=lr_decayed_fn, clipnorm=0.001
            )
        elif optimizer_str == 'SGD':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr_decayed_fn, clipnorm=0.001
            )
        
    print("optimizer", optimizer, "lr", lr)

    for batch_size in batch_size_list:
        # for progressive_unrolled_steps in range(unrolled_steps, unrolled_steps + 1):
        #c1 needs matching size, so we need to adjust it for every batch_size change!
        hw.c1 = math.tensor(np.repeat(c1["train"], batch_size), math.batch('batch'))
        # parameters_list[0] only used to get "dx" and "dt" so any input dataset should be correct
        loss_function = tf_loss.get_loss_function(
            progressive_unrolled_steps,
            # std_omega,
            std_omega_robin,
            # std_density,
            std_density_robin,
            std_phi,
            integration_step,
            parameters_list["train"][
                0
            ],  # parameters_list["train"] only used for "dx" and "dt" so it's ok to pass the "train" parameters explicitely
            corrector_step,
            cnn,
            temporal_causality_eps,
        )

        print(
            "DEBUG slice np_data_dict_indices_shuffled_list:",
            np_data_dict_indices_shuffled_list,
        )
        is_first_progressive_unrolled_steps = False

        min_data_length = get_min_data_length(
            np_data_dict_shuffled_unrolled_list, dataset_names
        )

        epochs = epochs_steps
        if (
            progressive_unrolled_steps != unrolled_steps
        ):  # 1% of the epochs for the smaller unrolled steps
            epochs = max(min(100, int(epochs / 10)), 100)

        # TRAINING
        n_steps_per_epoch = batch_size * (min_data_length["train"] // batch_size)
        print(n_steps_per_epoch)
        list_evaluation_metrics = []
        best_evaluation_metrics = {"valid":9999999.9}
        best_evaluation_metrics_file = None
        best_evaluation_metrics_epoch = 0
        start_training_time = time.time()
        MAX_HOURS=23
        for i_epoch in tqdm(range(0, epochs), disable=False, mininterval=10):
            elapsed_training_time = time.time() - start_training_time   #in secondes
            if elapsed_training_time >= MAX_HOURS*60*60:
                print("REACHED MAX HOURS:",  MAX_HOURS*60*60, elapsed_training_time)
                break
            # If min_data_length == batch_size each step has all the data, no need to shuffle
            if min_data_length["train"] != batch_size and (
                ((i_epoch * batch_size) % n_steps_per_epoch) == 0
            ):  # we finished one full pass through the data, we shuffle
                print("Step", i_epoch, "Epoch over, shuffle data")
                shuffle_dataset(
                    np_data_dict_shuffled_unrolled_list["train"],
                    np_data_dict_indices_shuffled_list["train"],
                )
                print(
                    "Shuffle np_data_dict_indices_shuffled_list:",
                    np_data_dict_indices_shuffled_list,
                )

            looped_index = (i_epoch * batch_size) % n_steps_per_epoch
            random_indices = np.arange(
                looped_index, looped_index + batch_size, dtype=int
            )
            print(
                "random_indices:",
                random_indices,
                "array indices",
                np.array(np_data_dict_indices_shuffled_list["train"][0])[
                    random_indices
                ],
            )
            print("free GPU mem before plasma_in:", get_gpu_memory())
            print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
            print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
            plasma_in = tf_npdict_to_tensordict.n_o_p_dict_concat(
                [
                    tf_npdict_to_tensordict.n_o_p_dict(
                        np_data_dict_shuffled_unrolled,
                        random_indices,
                        0,
                        batch_size,
                        parameters_list["train"][0]["dx"],
                    )
                    for np_data_dict_shuffled_unrolled in np_data_dict_shuffled_unrolled_list[
                        "train"
                    ]
                ]
            )
            # print("plasma_in:", plasma_in)
            # print("plasma_in.density.device:", plasma_in.density.device)
            print("free GPU mem after plasma_in:", get_gpu_memory())
            print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
            print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
            plasma_out_list = []
            for plasma_out_step in range(1, progressive_unrolled_steps + 1):
                plasma_out = tf_npdict_to_tensordict.n_o_p_dict_concat(
                    [
                        tf_npdict_to_tensordict.n_o_p_dict(
                            np_data_dict_shuffled_unrolled,
                            random_indices,
                            plasma_out_step,
                            batch_size,
                            parameters_list["train"][0]["dx"],
                        )
                        for np_data_dict_shuffled_unrolled in np_data_dict_shuffled_unrolled_list[
                            "train"
                        ]
                    ]
                )
                plasma_out_list.append(
                    [plasma_out.omega, plasma_out.density, plasma_out.phi]
                )
            # update_weights does the sum over the batch dimension. (actually over all dimensions still present in the loss)
            print("i_epoch:", i_epoch, "SAM_warmup:", SAM_warmup)
            if i_epoch<SAM_warmup or rho<=0.0:
                loss, pred = nn.update_weights(
                    cnn,
                    optimizer,
                    loss_function,
                    plasma_in,
                    plasma_out_list,  # all unrolled states for computing error
                )
            else:
                loss, pred = tf_update_weights.update_weights_SAM(
                    cnn,
                    optimizer,
                    loss_function,
                    rho,
                    plasma_in,
                    plasma_out_list,  # all unrolled states for computing error
                )
            print("free GPU mem after plasma_in:", get_gpu_memory())
            # print("cnn.losses", cnn.losses)
            print("loss:", loss)
            losses_list["train"].append(
                math.mean(loss, dim=math.batch("batch")).numpy()
            )
            if weight_average > 0:
                if (
                    i_epoch > epochs - weight_average - 1
                ):  # We are in the last steps where model averaging is performed
                    tf_tools.weight_averaging(
                        cnn,  # current model
                        output_path
                        + "_models/model_unrolled"
                        + str(progressive_unrolled_steps)
                        + "_average",
                        # + ".keras",  # file name of the saved average model
                        i_epoch
                        - (
                            epochs - weight_average - 1
                        ),  # number of average models in the previously saved file.
                    )
                if (
                    i_epoch >= epochs - weight_average - 1
                ):  # One epochbefore model averaging is performed we start saving the current averaged model.
                    cnn.save(
                        output_path
                        + "_models/model_unrolled"
                        + str(progressive_unrolled_steps)
                        + "_average",
                        # + ".keras",
                        save_format='tf',
                    )

            if (i_epoch + 1) % save_model_skips == 0:
                print(
                    "save model:",
                    output_path
                    + "_models/model_unrolled"
                    + str(progressive_unrolled_steps)
                    + "_epoch"
                    + str(i_epoch),
                    # + ".keras",
                )
                cnn.save(
                    output_path
                    + "_models/model_unrolled"
                    + str(progressive_unrolled_steps)
                    + "_epoch"
                    + str(i_epoch),
                    # + ".keras",
                    save_format='tf',
                )
            # VALID
            if (i_epoch + 1) % valid_skips == 0 and (i_epoch + 1)>valid_warmup:
                save_name_eval = (
                    output_path
                    + "_models/model_unrolled"
                    + str(progressive_unrolled_steps)
                    + "_epoch"
                    + "_"
                    + str(batch_size)
                    + "midrun"
                    + "_"
                    + str(i_epoch)
                    # + ".keras"
                )
                cnn.save(save_name_eval, save_format='tf')
                print("Debug: now valid and test")
                total_time_convert_tensorrt = time.time()
                if tensorrt_inference:
                    cnn_converted_tensorrt, converter, model_converted = convert_model_to_tensorrt(["valid","test"], valid_batch, save_name_eval, parameters_list["train"][0]["grid_pts"])
                    cnn_inference = lambda x : list(cnn_converted_tensorrt(tf.convert_to_tensor(x)).values())[0] #We need to convert_to_tensor in case we are in math.use("numpy") and the array will be a numpy array. It returns a dictionnary so we have to extract the output
                else:
                    # cnn_inference = cnn
                    cnn_inference = tf.keras.models.load_model(
                        save_name_eval
                    )  # to make sure it doesn't change any training values, is it necessary?
                print("total_time_convert_tensorrt:", time.time() - total_time_convert_tensorrt)
                evaluation_metrics = evaluation_test(
                    np_data_dict_list,
                    np_data_dict_indices_shuffled_list,
                    ["valid", "test"],
                    parameters_list,
                    output_path + "_" + str(batch_size) + "Ep" + str(i_epoch),
                    buffer_size,
                    continue_file,
                    physics_params,
                    snaps,
                    steps_simulation, 
                    progressive_unrolled_steps,  #divide by progressive_unrolled_steps since we decimated the data by that amount
                    solver_name,
                    hw,
                    corrector_step,
                    cnn_inference,   
                    [
                        "gamma_n_spectral",
                        #"gamma_c",
                        "thermal_energy",
                        "kinetic_energy",
                        "enstrophy",
                    ],
                    ["thermal_energy", "kinetic_energy", "enstrophy"],
                    movie,
                    min_fps,
                    dpi,
                    speed,
                    {"valid": valid_dataset_path, "test": test_dataset_path},
                    c1,
                    valid_batch,
                    evaluation_properties,
                    n_blocks_invariants,
                    keep_all_simulation_data,
                )
                list_evaluation_metrics.append(evaluation_metrics)
                if evaluation_metrics["valid"] < best_evaluation_metrics["valid"]:
                    best_evaluation_metrics = evaluation_metrics
                    best_evaluation_metrics_file = save_name_eval
                    best_evaluation_metrics_epoch = i_epoch
                
                if not np.isfinite(evaluation_metrics["valid"]):
                    print("VALID METRIC NOT FINITE")
                    break
                if early_stopping_patience>1:
                    if len(list_evaluation_metrics) > early_stopping_patience:
                        window_best_evaluation_metrics_valid = np.min([best_met["valid"] for best_met in list_evaluation_metrics[-early_stopping_patience:]])
                        if window_best_evaluation_metrics_valid>best_evaluation_metrics["valid"]*1.05: #with 5% margin
                            print("EARLY STOPPING 5%margin patience", early_stopping_patience, "best", best_evaluation_metrics["valid"], "lasts", window_best_evaluation_metrics_valid)
                            break


                #WE DON'T VALIDATE WITH THE TRAINING UNROLLED STEPS ANYMORE
                # for d_name in ["valid", "test"]:
                #     tmp_losses_valid = []
                #     # Should we shuffle the validation dataset or not???
                #     # shuffle_dataset(
                #     #     np_data_dict_shuffled_unrolled_list[d_name],
                #     #     np_data_dict_indices_shuffled_list[d_name],
                #     # )
                #     for j_valid in range(0, valid_epochs):
                #         print(d_name, j_valid)
                #         looped_index = (j_valid * batch_size) % max(
                #             1, min_data_length[d_name] - batch_size
                #         )
                #         random_indices = np.arange(
                #             looped_index, looped_index + batch_size, dtype=int
                #         )
                #         print(
                #             d_name,
                #             [
                #                 [l[r] for r in random_indices]
                #                 for l in np_data_dict_indices_shuffled_list[d_name]
                #             ],
                #         )
                #         plasma_in_valid = tf_npdict_to_tensordict.n_o_p_dict_concat(
                #             [
                #                 tf_npdict_to_tensordict.n_o_p_dict(
                #                     np_data_dict_shuffled_unrolled,
                #                     random_indices,
                #                     0,
                #                     batch_size,
                #                     parameters_list[d_name][0]["dx"],
                #                 )
                #                 for np_data_dict_shuffled_unrolled in np_data_dict_shuffled_unrolled_list[
                #                     d_name
                #                 ]
                #             ]
                #         )
                #         # ground truth
                #         plasma_out_list_valid = []
                #         for plasma_out_step in range(1, progressive_unrolled_steps + 1):
                #             plasma_out = tf_npdict_to_tensordict.n_o_p_dict_concat(
                #                 [
                #                     tf_npdict_to_tensordict.n_o_p_dict(
                #                         np_data_dict_shuffled_unrolled,
                #                         random_indices,
                #                         plasma_out_step,
                #                         batch_size,
                #                         parameters_list[d_name][0]["dx"],
                #                     )
                #                     for np_data_dict_shuffled_unrolled in np_data_dict_shuffled_unrolled_list[
                #                         d_name
                #                     ]
                #                 ]
                #             )
                #             plasma_out_list_valid.append(
                #                 [plasma_out.omega, plasma_out.density, plasma_out.phi]
                #             )
                #     #     loss_valid, pred_valid = loss_function(
                #     #         plasma_in_valid,
                #     #         plasma_out_list_valid,  # all unrolled states for computing error
                #     #     )
                #     #     print("loss", d_name, loss_valid)
                #     #     tmp_losses_valid.append(
                #     #         math.mean(loss_valid, dim=math.batch("batch")).numpy()
                #     #     )
                #     # losses_list[d_name].append(np.mean(tmp_losses_valid))

    if epochs_steps > 0:
        print(
            "save model:",
            output_path
            + "_models/model_unrolled"
            + str(progressive_unrolled_steps)
            + "_epoch"
            + str(i_epoch),
            # + ".keras",
        )
        cnn.save(
            output_path
            + "_models/model_unrolled"
            + str(progressive_unrolled_steps)
            + "_epoch"
            + str(i_epoch),
            # + ".keras",
            save_format='tf',
        )
        # print("cnn.weights", cnn.weights)
        print("loss:", loss)
        print("pred:", pred)
        print("losses_list", losses_list)

    # SAVE ALL
    # initial plasma

    save_name_eval = (
        output_path
        + "_models/model_unrolled"
        + str(progressive_unrolled_steps)
        + "_epoch"
        + "_"
        + str(batch_size)
        + "midrun"
        + "_"
        + str(i_epoch)
        #+ ".keras"
    )
    cnn.save(save_name_eval, save_format='tf')
    #CONVERT MODEL TO TENSORRT FOR FAST INFERENCE!!!
    #We need to return the converter and(? maybe just one of the two, haven't checked) model_converted. Otherwise performance becomes 50X worse!!!
    total_time_convert_tensorrt = time.time()
    if tensorrt_inference:
        cnn_converted_tensorrt, converter, model_converted = convert_model_to_tensorrt(["valid","test"], valid_batch, save_name_eval, parameters_list["train"][0]["grid_pts"])
        cnn_inference = lambda x : list(cnn_converted_tensorrt(tf.convert_to_tensor(x)).values())[0] #We need to convert_to_tensor in case we are in math.use("numpy") and the array will be a numpy array. It returns a dictionnary so we have to extract the output
    else:
        # cnn_inference = cnn
        cnn_inference = tf.keras.models.load_model(
            save_name_eval
        )  # to make sure it doesn't change any training values, is it necessary?
    print("total_time_convert_tensorrt:", time.time() - total_time_convert_tensorrt)
    evaluation_metrics = evaluation_test(
        np_data_dict_list,
        np_data_dict_indices_shuffled_list,
        ["valid","test"],
        parameters_list,
        output_path + "_firsttest",
        buffer_size,
        continue_file,
        physics_params,
        snaps,
        steps_simulation,
        progressive_unrolled_steps,  #divide by progressive_unrolled_steps since we decimated the data by that amount
        solver_name,
        hw,
        corrector_step,
        cnn_inference,
        [
            "gamma_n_spectral",
            #"gamma_c",
            "thermal_energy",
            "kinetic_energy",
            "enstrophy",
        ],
        ["thermal_energy", "kinetic_energy", "enstrophy"],
        movie,
        min_fps,
        dpi,
        speed,
        {"valid": valid_dataset_path, "test": test_dataset_path},
        c1,
        valid_batch,
        evaluation_properties,
        n_blocks_invariants,
        keep_all_simulation_data,
    )
        
    list_evaluation_metrics.append(evaluation_metrics)
    if evaluation_metrics["valid"] < best_evaluation_metrics["valid"]:
        best_evaluation_metrics = evaluation_metrics
        best_evaluation_metrics_file = save_name_eval
        best_evaluation_metrics_epoch = i_epoch
    
    print("(epoch) best_evaluation_metrics:", best_evaluation_metrics_epoch, best_evaluation_metrics)
    print("best_evaluation_metrics_file:", best_evaluation_metrics_file)




def load_h5py_data(
    train_dataset_path, valid_dataset_path, test_dataset_path, dataset_names
):
    print("train_dataset_path:", train_dataset_path)
    h5_file_dataset_list = {
        "train": [h5py.File(d_path, "r") for d_path in train_dataset_path],
        "valid": [h5py.File(d_path, "r") for d_path in valid_dataset_path],
        "test": [h5py.File(d_path, "r") for d_path in test_dataset_path],
    }
    print("train_dataset_path opened")
    # parameters = {'N': 3, 'arakawa_coeff': 1.0, 'c1': 1.0, 'dt': 0.07500000000000001, 'dx': 4.188790204786391, 'frame_dt': 0.07500000000000001, 'grid_pts': 10, 'k0': 0.15, 'kappa_coeff': 1.0, 'nu': 5e-08, 'x': 10, 'y': 10}
    parameters_list = {
        key_set: [
            dict(h5_file_dataset.attrs)
            for h5_file_dataset in h5_file_dataset_list[key_set]
        ]
        for key_set in dataset_names
    }
    # ONLY NEED ONE FRAME FOR INPUT.
    # IT'S THE OUTPUT THAT NEEDS MULTIPLE FRAMES DEPENDING ON THE UNROLLING.
    np_data_dict_list = {
        "train": [{} for _ in train_dataset_path],
        "valid": [{} for _ in valid_dataset_path],
        "test": [{} for _ in test_dataset_path],
    }
    print("load arrays")
    for key_set in dataset_names:
        for np_data_dict, h5_file_dataset in zip(
            np_data_dict_list[key_set], h5_file_dataset_list[key_set]
        ):
            for field in h5_file_dataset.keys():
                print(field)
                np_data_dict[field] = np.array(h5_file_dataset[field])
            h5_file_dataset.close()
    print("loaded arrays")
    return parameters_list, np_data_dict_list


def extract_windows_vectorized(
    array, sub_window_size, stride
):  # extracts sliding window for axis=0 and place the slide_windows at axis=-1
    max_time = array.shape[0] - sub_window_size
    sub_windows = (
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(sub_window_size), 0)
        + np.expand_dims(np.arange(0, max_time + 1, stride), 0).T
    )
    return np.moveaxis(array[sub_windows], 1, -1)


def get_data_mean_std_robin(np_data_dict_list):
    mean_omega = np.mean(
        [np.mean(np_data_dict["omega"]) for np_data_dict in np_data_dict_list]
    )
    std_omega = np.mean(
        [np.std(np_data_dict["omega"]) for np_data_dict in np_data_dict_list]
    )
    mean_density = np.mean(
        [np.mean(np_data_dict["density"]) for np_data_dict in np_data_dict_list]
    )
    std_density = np.mean(
        [np.std(np_data_dict["density"]) for np_data_dict in np_data_dict_list]
    )
    mean_phi = np.mean(
        [np.mean(np_data_dict["phi"]) for np_data_dict in np_data_dict_list]
    )
    std_phi = np.mean(
        [np.std(np_data_dict["phi"]) for np_data_dict in np_data_dict_list]
    )
    return mean_omega, std_omega, mean_density, std_density, mean_phi, std_phi


def get_data_mean_std_slices(np_data_dict_list):
    [
        print("np_data_dict[omega].shape", np_data_dict["omega"].shape)
        for np_data_dict in np_data_dict_list
    ]
    mean_omega = np.mean(
        np.concatenate(
            [
                np.mean(
                    np_data_dict["omega"],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    std_omega = np.mean(
        np.concatenate(
            [
                np.std(
                    np_data_dict["omega"],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    mean_density = np.mean(
        np.concatenate(
            [
                np.mean(
                    np_data_dict["density"],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    std_density = np.mean(
        np.concatenate(
            [
                np.std(
                    np_data_dict["density"],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    mean_phi = np.mean(
        np.concatenate(
            [
                np.mean(
                    np_data_dict["phi"],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    std_phi = np.mean(
        np.concatenate(
            [
                np.std(
                    np_data_dict["phi"],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    return mean_omega, std_omega, mean_density, std_density, mean_phi, std_phi


def get_data_mean_std_slices_diff(np_data_dict_list):
    mean_omega = np.mean(
        np.concatenate(
            [
                np.mean(
                    np_data_dict["omega"][1:, ...] - np_data_dict["omega"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    std_omega = np.mean(
        np.concatenate(
            [
                np.std(
                    np_data_dict["omega"][1:, ...] - np_data_dict["omega"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                + 1e-6
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    mean_density = np.mean(
        np.concatenate(
            [
                np.mean(
                    np_data_dict["density"][1:, ...]
                    - np_data_dict["density"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    std_density = np.mean(
        np.concatenate(
            [
                np.std(
                    np_data_dict["density"][1:, ...]
                    - np_data_dict["density"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                + 1e-6
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    mean_phi = np.mean(
        np.concatenate(
            [
                np.mean(
                    np_data_dict["phi"][1:, ...] - np_data_dict["phi"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    std_phi = np.mean(
        np.concatenate(
            [
                np.std(
                    np_data_dict["phi"][1:, ...] - np_data_dict["phi"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                + 1e-6
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    return mean_omega, std_omega, mean_density, std_density, mean_phi, std_phi


def get_data_mean_std_slices_diff_hw(np_data_dict_list, integration_step, parameters):
    mean_omega = np.mean(
        np.concatenate(
            [
                np.mean(
                    np_data_dict["omega"][1:, ...] - np_data_dict["omega"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    std_omega = np.mean(
        np.concatenate(
            [
                np.std(
                    np_data_dict["omega"][1:, ...] - np_data_dict["omega"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                + 1e-6
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    mean_density = np.mean(
        np.concatenate(
            [
                np.mean(
                    np_data_dict["density"][1:, ...]
                    - np_data_dict["density"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    std_density = np.mean(
        np.concatenate(
            [
                np.std(
                    np_data_dict["density"][1:, ...]
                    - np_data_dict["density"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                + 1e-6
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    mean_phi = np.mean(
        np.concatenate(
            [
                np.mean(
                    np_data_dict["phi"][1:, ...] - np_data_dict["phi"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    std_phi = np.mean(
        np.concatenate(
            [
                np.std(
                    np_data_dict["phi"][1:, ...] - np_data_dict["phi"][:-1, ...],
                    axis=(
                        1,
                        2,
                    ),
                )
                + 1e-6
                for np_data_dict in np_data_dict_list
            ]
        )
    )
    return mean_omega, std_omega, mean_density, std_density, mean_phi, std_phi


def shuffle_dataset(dataset, indices):
    for (
        i_index_list,
        np_data_dict_shuffled_unrolled,
    ) in enumerate(dataset):
        st0 = np.random.get_state()
        np.random.shuffle(indices[i_index_list])
        for k in np_data_dict_shuffled_unrolled.keys():
            np.random.set_state(st0)  # to shufle the indicies exactly the same way
            np.random.shuffle(np_data_dict_shuffled_unrolled[k])
            print("shuffle", k, np_data_dict_shuffled_unrolled[k].shape)


def get_min_data_length(np_data_dict_shuffled_unrolled_list, dataset_names):
    min_data_length = {
        key_set: np_data_dict_shuffled_unrolled_list[key_set][0]["density"].shape[0]
        for key_set in dataset_names
    }
    for key_set in dataset_names:
        for np_data_dict_shuffled_unrolled in np_data_dict_shuffled_unrolled_list[
            key_set
        ]:
            min_data_length[key_set] = min(
                min_data_length[key_set],
                np_data_dict_shuffled_unrolled["density"].shape[0],
            )
    print("min_data_length:", min_data_length)
    return min_data_length


def sort_and_extract_windows_vectorized(
    dataset_names,
    np_data_dict_shuffled_unrolled_list,
    is_first_progressive_unrolled_steps,
    np_data_dict_indices_shuffled_list,
    progressive_unrolled_steps,
):
    for key_set in dataset_names:
        for i_datalist, np_data_dict_shuffled_unrolled in enumerate(
            np_data_dict_shuffled_unrolled_list[key_set]
        ):
            if (
                not is_first_progressive_unrolled_steps
            ):  # It's allready unrolled! (Batch, Y, X, Unrolled)
                for k in np_data_dict_shuffled_unrolled.keys():
                    # we first de-roll it
                    # This looses the last slices!!! the last unrolled slices of the last example are now lost
                    # negligeable if we have a lot of data and small unrolled sizes
                    # we use np_data_dict_indices_shuffled_list[key_set][i_datalist] to sort back the data
                    print(
                        key_set,
                        k,
                        "indices to sort slices",
                        np_data_dict_indices_shuffled_list[key_set][i_datalist],
                        "argsort",
                        np.argsort(
                            np_data_dict_indices_shuffled_list[key_set][i_datalist]
                        ),
                    )
                    # [6009, 6003, 6004, 6007, 6001, 6006, 6002, 6008, 6000, 6005]
                    np_data_dict_shuffled_unrolled[k] = np_data_dict_shuffled_unrolled[
                        k
                    ][
                        np.argsort(
                            np_data_dict_indices_shuffled_list[key_set][i_datalist]
                        ),
                        ...,
                        0,
                    ]  # substract the min of the indices since the data is truncated from the first 30% so the first index is still at 30%
                np_data_dict_indices_shuffled_list[key_set][
                    i_datalist
                ].sort()  # We are back to an unsorted array
                print(
                    key_set,
                    "indices sorted",
                    np_data_dict_indices_shuffled_list[key_set][i_datalist],
                )
            np_data_dict_indices_shuffled_list[key_set][
                i_datalist
            ] = np_data_dict_indices_shuffled_list[key_set][i_datalist][
                :-progressive_unrolled_steps
            ]
            for k in np_data_dict_shuffled_unrolled.keys():
                print("before window", np_data_dict_shuffled_unrolled[k].shape)
                np_data_dict_shuffled_unrolled[k] = extract_windows_vectorized(
                    np_data_dict_shuffled_unrolled[k],
                    progressive_unrolled_steps + 1,
                    progressive_unrolled_steps
                )
                print("after window", np_data_dict_shuffled_unrolled[k].shape)
                print(
                    "window",
                    key_set,
                    k,
                    np_data_dict_shuffled_unrolled[k].shape,
                    np_data_dict_indices_shuffled_list[key_set][i_datalist].shape,
                )
                #DECIMATE to keep memory to reasonnable levels!
                #np_data_dict_shuffled_unrolled[k] = np_data_dict_shuffled_unrolled[k][::progressive_unrolled_steps, ...]   #NOT NEEDED ANYMORE, THE extract_windows_vectorized FUNCTION HAS A NEW STRIDE PARAMETER
                print("after DECIMATE", np_data_dict_shuffled_unrolled[k].shape,np_data_dict_indices_shuffled_list[key_set][i_datalist].shape,)
                print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
            #DECIMATE
            np_data_dict_indices_shuffled_list[key_set][
                i_datalist
            ] = np_data_dict_indices_shuffled_list[key_set][i_datalist][
                :np_data_dict_shuffled_unrolled[k].shape[0],...
            ]


def get_initial_test_plasma(np_data_dict_list, dataset_list, parameters_list, steps_simulation, valid_batch):
    # deep copy so that we don't modify the test data directly
    # initial plasma
    # All files are used. One sample (the first) per file.
    random_indices = slice(0, 1, 1)
    # print("np_data_dict_list[dataset_list]:", np_data_dict_list[dataset_list])
    print(
        "np_data_dict_list[dataset_list[0]][0]['density'].shape:",
        np_data_dict_list[dataset_list[0]][0]["density"].shape[0],
    )
    length_simulation = np_data_dict_list[dataset_list[0]][0]["density"].shape[0]
    # (length_simulation-steps_simulation) so that the last newly simulated point is still comparable to an existing point in the ground truth
    random_indices = np.linspace(0, np.maximum(0, length_simulation-steps_simulation), valid_batch, dtype=int)
    n_indices = random_indices.shape[0]
    return (
        copy.deepcopy(
            tf_npdict_to_tensordict.n_o_p_dict_concat(
                [
                    tf_npdict_to_tensordict.n_o_p_dict(
                        np_data_dict_shuffled_unrolled,
                        random_indices,
                        0,
                        n_indices,
                        parameters_list[dataset][0]["dx"],
                    )
                    for dataset in dataset_list
                    for np_data_dict_shuffled_unrolled in np_data_dict_list[dataset]
                ]
            )
        ),
        n_indices,
        random_indices,
        [len(np_data_dict_list[dataset]) for dataset in dataset_list],
    )


def set_save_file_handling(
    output_path,
    buffer_size,
    parameters_list,
    continue_file,
    physics_params,
    snaps,
    plasma,
    batch_index,
):
    field_list = ("density", "omega", "phi")
    dim_order = ("batch", "y", "x", "gradient")
    if output_path:
        buffer = {
            field: np.zeros(
                (
                    buffer_size,
                    parameters_list["test"][0]["y"],
                    parameters_list["test"][0]["x"],
                ),
                dtype=np.float32,
            )
            for field in field_list
        }
        output_params = {
            "buffer": buffer,
            "buffer_index": 0,
            "output_path": output_path,
        }
        # Load Data
        if os.path.isfile(output_path):
            if continue_file:
                plasma, physics_params = continue_h5_file(output_path, field_list)
                print(
                    f"Successfully loaded: {output_path} (age={plasma.age})\n{physics_params}"
                )
            else:
                print(f"File already exists.")
                return
        # Create
        else:
            save_params = get_save_params(
                physics_params,
                parameters_list["test"][0]["dt"],
                snaps,
                parameters_list["test"][0]["y"],
                parameters_list["test"][0]["x"],
            )
            create_appendable_h5(
                output_path,
                save_params,
                chunk_size=100,
            )
            new_p = math.Dict(
                density=plasma.density.batch[batch_index : batch_index + 1],
                omega=plasma.omega.batch[batch_index : batch_index + 1],
                phi=plasma.phi.batch[batch_index : batch_index + 1],
                age=0,
                dx=plasma.dx,
            )
            # print("plasma.density", plasma.density)
            # print("new_p.density", new_p.density)
            output_params["buffer_index"] = save_to_buffered_h5(
                # new_val=plasma,
                new_val=new_p,
                buffer_size=buffer_size,
                dim_order=dim_order,
                **output_params,
            )
        return field_list, dim_order, output_params, plasma, physics_params


def run_simulation(
    steps_simulation,
    solver_name,
    hw,
    plasma,
    parameters_list,
    corrector_step,
    cnn,
    list_output_path,
    snaps,
    buffer_size,
    list_file_handling,
):
    print("Running simulation...")
    numba_plasma = {}
    total_time_data = 0.0
    total_time_euler = 0.0
    total_time_nn = 0.0
    for i in tqdm(
        range(1, steps_simulation), disable=False, mininterval=10
    ):  # 2000)):  # 16000)):
        # Progress one step, alternatively: hw.euler_step()
        if False:
            if solver_name == "euler":
                plasma = hw.euler_step(
                    plasma,
                    dt=parameters_list["test"][0]["dt"],
                    dx=parameters_list["test"][0]["dx"],
                )
            elif solver_name == "rk2":
                plasma = hw.rk2_step(
                    plasma,
                    dt=parameters_list["test"][0]["dt"],
                    dx=parameters_list["test"][0]["dx"],
                )
            elif solver_name == "rk4":
                plasma = hw.rk4_step(
                    plasma,
                    dt=parameters_list["test"][0]["dt"],
                    dx=parameters_list["test"][0]["dx"],
                )
        else:
            t0 = time.time()
            dim_order = ("batch", "y", "x", "gradient")
            n_dims = plasma["density"].rank
            numba_plasma["density"] = plasma["density"].numpy(dim_order[:n_dims])
            numba_plasma["omega"] = plasma["omega"].numpy(dim_order[:n_dims])
            numba_plasma["phi"] = plasma["phi"].numpy(dim_order[:n_dims])
            numba_plasma["age"] = plasma["age"]
            numba_plasma["dx"] = plasma["dx"]
            numba_plasma = Namespace(
                density=plasma["density"].numpy(dim_order[:n_dims]),
                omega=plasma["omega"].numpy(dim_order[:n_dims]),
                phi=plasma["phi"].numpy(dim_order[:n_dims]),
                age=plasma["age"],
                dx=plasma["dx"],
            )
            total_time_data = total_time_data + time.time() - t0
            t1 = time.time()
            if solver_name == "euler":
                numba_plasma = hw.euler_step(
                    numba_plasma,
                    dt=parameters_list["test"][0]["dt"],
                    dx=parameters_list["test"][0]["dx"],
                )
            elif solver_name == "rk2":
                numba_plasma = hw.rk2_step(
                    numba_plasma,
                    dt=parameters_list["test"][0]["dt"],
                    dx=parameters_list["test"][0]["dx"],
                )
            elif solver_name == "rk4":
                numba_plasma = hw.rk4_step(
                    numba_plasma,
                    dt=parameters_list["test"][0]["dt"],
                    dx=parameters_list["test"][0]["dx"],
                )
            total_time_euler = total_time_euler + time.time() - t1
            t0 = time.time()
            plasma["density"] = math.tensor(
                numba_plasma["density"], math.batch("batch"), math.spatial("y", "x")
            )
            plasma["omega"] = math.tensor(
                numba_plasma["omega"], math.batch("batch"), math.spatial("y", "x")
            )
            plasma["phi"] = math.tensor(
                numba_plasma["phi"], math.batch("batch"), math.spatial("y", "x")
            )
            plasma["age"] = numba_plasma["age"]
            plasma["dx"] = numba_plasma["dx"]
            total_time_data = total_time_data + time.time() - t0

        t2 = time.time()
        plasma = corrector_step(plasma, cnn)
        total_time_nn = total_time_nn + time.time() - t2
        for batch_index, (
            output_path,
            (
                field_list,
                dim_order,
                output_params,
                _,
                physics_params,
            ),
        ) in enumerate(zip(list_output_path, list_file_handling)):
            # Save to records
            if output_path and i % snaps == 0:
                new_p = math.Dict(
                    density=plasma.density.batch[batch_index : batch_index + 1],
                    omega=plasma.omega.batch[batch_index : batch_index + 1],
                    phi=plasma.phi.batch[batch_index : batch_index + 1],
                    age=0,
                    dx=plasma.dx,
                )
                output_params["buffer_index"] = save_to_buffered_h5(
                    # new_val=plasma,
                    new_val=new_p,
                    buffer_size=buffer_size,
                    dim_order=dim_order,
                    **output_params,
                )

        # Check for breaking
        if math.is_nan(plasma.density).any:
            print(f"FAILED @ {i:,} steps ({plasma.age:,})")
            break
        # if i%1000 == 0:
        #    fig, ax = plt.subplots(1, 3)
        #    ax[0].matshow(plasma.density)
        #    ax[1].matshow(plasma.omega)
        #    ax[2].matshow(plasma.phi)
        #    plt.show()
    # If output_path is defined, flush any remaining data in the buffer
    print("total_time_data:", total_time_data)
    print("total_time_euler:", total_time_euler)
    print("total_time_nn:", total_time_nn)
    for output_path, (
        field_list,
        dim_order,
        output_params,
        _,
        physics_params,
    ) in zip(list_output_path, list_file_handling):
        if output_path and output_params["buffer_index"] > 0:
            append_h5(**output_params)


def compute_properties_make_plots_and_movie(
    properties,
    output_path,
    buffer_size,
    plot_properties,
    movie,
    field_list,
    min_fps,
    dpi,
    speed,
):

    t0 = time.time()
    if properties and output_path:
        print(f"Calculating properties...")
        calculate_properties(
            file_path=output_path,
            batch_size=buffer_size,
            property_list=properties,
            force_recompute=True,
            is_debug=False,
        )
    total_time_calculate_properties = time.time() - t0

    t0 = time.time()
    if plot_properties and output_path:
        print(f"Plotting properties...")
        plot_timetraces(
            file_path=output_path,
            out_path=None,
            properties=plot_properties,
            t0=0,
            t0_std=0,
        )
        plot_timetraces(
            file_path=output_path,
            out_path=None,
            properties=("gamma_n_spectral",),#, "gamma_c"),
            t0=0,
            t0_std=0,
        )
    total_time_timetraces = time.time() - t0
    print("total_time_calculate_properties:", total_time_calculate_properties)
    print("total_time_timetraces:", total_time_timetraces)
    # Generate Movie from saved file
    if movie and output_path:
        print(f"Generating movie...")
        create_movie(
            input_filename=output_path,
            output_filename=output_path,
            t0=0,
            t1=None,
            plot_order=field_list,
            min_fps=min_fps,
            dpi=dpi,
            speed=speed,
        )


def get_invariant_moments(file_name, start_index=0, end_index=None, evaluation_properties=("gamma_n_spectral",)):
    h5_file_dataset = h5py.File(file_name, "r")
    list_properties = {}
    for evaluation_prop in evaluation_properties:
        if end_index is None:
            list_properties[evaluation_prop] = math.tensor(np.array(h5_file_dataset[evaluation_prop]), math.batch("time"))
        else:
            list_properties[evaluation_prop] = math.tensor(np.array(h5_file_dataset[evaluation_prop][start_index:end_index]), math.batch("time"))
    h5_file_dataset.close()

    for evaluation_prop in evaluation_properties:
        print(
            file_name,
            evaluation_prop,
            "g",
            "mean std",
            math.mean(list_properties[evaluation_prop], dim="time"),
            math.std(list_properties[evaluation_prop], dim="time"),
        )
    return [e for evaluation_prop in evaluation_properties for e in [math.mean(list_properties[evaluation_prop], dim="time").numpy(), math.std(list_properties[evaluation_prop], dim="time").numpy()]]



def get_invariant_block_moments(file_name, start_index=0, end_index=None, evaluation_properties=("gamma_n_spectral",), n_blocks_invariants=4):
    h5_file_dataset = h5py.File(file_name, "r")
    list_properties = {}
    for evaluation_prop in evaluation_properties:
        if end_index is None:
            list_properties[evaluation_prop] = math.tensor(np.array(h5_file_dataset[evaluation_prop]), math.batch("time"))
        else:
            list_properties[evaluation_prop] = math.tensor(np.array(h5_file_dataset[evaluation_prop][start_index:end_index]), math.batch("time"))
    h5_file_dataset.close()

    for evaluation_prop in evaluation_properties:
        print(
            file_name,
            evaluation_prop,
            "g",
            "mean std",
            block_mean_std(list_properties[evaluation_prop], n_blocks_invariants, dim="time"),
        )
    return [e for evaluation_prop in evaluation_properties for e in block_mean_std(list_properties[evaluation_prop], n_blocks_invariants, dim="time")]


def block_mean_std(time_series, n_blocks_invariants, dim):
    n_blocks_invariants = n_blocks_invariants + 1   #we add one block at the beginning that we then ignore, to remove the first part where the simulations are corellated
    length_time_series = time_series.time.size
    size_block = length_time_series//n_blocks_invariants
    left_over = length_time_series % n_blocks_invariants
    block_means = []
    #block_means = [math.mean(time_series.time[:left_over+size_block], dim=dim)]    #we skip the first block
    for i in range(1, n_blocks_invariants):
        block_means.append(math.mean(time_series.time[left_over+i*size_block:left_over+(i+1)*size_block], dim=dim))
    print("block_means:", block_means)
    return [math.mean(block_means).numpy(), math.std(block_means).numpy()]


def evaluation_test(
    np_data_dict_list,
    np_data_dict_indices_shuffled_list,
    dataset_list,
    parameters_list,
    output_path,
    buffer_size,
    continue_file,
    physics_params,
    snaps,
    steps_simulation,
    progressive_unrolled_steps,
    solver_name,
    hw,
    corrector_step,
    cnn,
    properties,
    plot_properties,
    movie,
    min_fps,
    dpi,
    speed,
    test_dataset_path,
    c1,
    valid_batch,
    evaluation_properties,
    n_blocks_invariants,
    keep_all_simulation_data,
):
    math.use("numpy")
    # SAVE ALL
    # initial plasma
    (
        plasma,
        n_batches,
        plasma_start_index,
        list_n_files_per_datasets,
    ) = get_initial_test_plasma(np_data_dict_list, dataset_list, parameters_list, steps_simulation//progressive_unrolled_steps, valid_batch)   #divide by progressive_unrolled_steps since we decimated the data by that amount
    print("plasma", plasma, "plasma_start_index", plasma_start_index)
    numba_hw = numba_HW(**physics_params, debug=hw.debug)
    numba_hw.poisson_bracket_coeff = (
        -numba_hw.poisson_bracket_coeff
    )  # BECAUSE OF ROBIN'S HW2D ERROR WHERE THE POISSON BRACKET HAS SWITCHED X,Y AXIS!!! ONCE IT IS CORRECTED WE HAVE TO REMOVE THAT!!
    #numba_hw.c1 = np.repeat(numba_hw.c1.numpy().reshape(-1, 1, 1), n_batches, axis=0)    #so that we can multiply a field (batch, y, x) by per-batch c1 values (batch)
    list_c1 = [c1[k] for k in  dataset_list]
    print("list_c1:", list_c1)
    list_c1_flat = [ee for e in list_c1 for ee in e]
    print("list_c1_flat:", list_c1_flat)
    numba_hw.c1 = np.repeat(list_c1_flat, n_batches).reshape(-1, 1, 1)
    print("numba_hw.c1:", numba_hw.c1)
    # save_c1_training = hw.c1
    # hw.c1 = math.tensor(numba_hw.c1[:, 0, 0], math.batch(batch=len(numba_hw.c1)))
    
    # File Handling
    list_output_path = []
    list_file_handling = []
    index_plasma_batch = 0
    for dataset in dataset_list:
        n_files = len(parameters_list[dataset]) * n_batches
        for i in range(n_files):
            list_output_path.append(output_path + "_" + dataset + str(i))

        for _ in range(n_files):
            # (
            #     field_list,
            #     dim_order,
            #     output_params,
            #     plasma,
            #     physics_params,
            # ) = set_save_file_handling(
            list_file_handling.append(
                set_save_file_handling(
                    # output_path,
                    list_output_path[index_plasma_batch],
                    buffer_size,
                    parameters_list,
                    continue_file,
                    physics_params,
                    snaps,
                    plasma,
                    index_plasma_batch,
                )
            )
            index_plasma_batch = index_plasma_batch + 1

    # Run Simulation
    run_simulation(
        steps_simulation,
        solver_name,
        # hw,
        numba_hw,
        plasma,
        parameters_list,
        corrector_step,
        cnn,
        # output_path,
        list_output_path,
        snaps,
        buffer_size,
        list_file_handling,
    )
    math.use("tensorflow")
    
    t0 = time.time()
    # MAKE PLOTS
    for output_path, (
        field_list,
        _,
        _,
        _,
        _,
    ) in zip(list_output_path, list_file_handling):
        compute_properties_make_plots_and_movie(
            properties,
            output_path,
            buffer_size,
            plot_properties,
            # movie,
            False,  # No movie
            field_list,
            min_fps,
            dpi,
            speed,
        )
    total_time_properties_plot = time.time() - t0
    list_error_invariance = []
    t0 = time.time()
    index_plasma_batch = 0
    total_time_get_invariant = 0.0
    for dataset in dataset_list:
        # i:0, file_0
        # i:0, file_1
        # i:0, file_2
        # i:1, file_0
        # i:1, file_1
        # i:1, file_2
        for i, output_path in zip(
            np.arange(len(parameters_list[dataset]), dtype=np.int64).repeat(n_batches),
            list_output_path,
        ):
            t1 = time.time()
            start_index_ground_truth = np.min(np_data_dict_indices_shuffled_list[dataset][i]) + plasma_start_index[index_plasma_batch%n_batches]
            end_index_ground_truth = start_index_ground_truth + steps_simulation
            # test_ground_truth_invariant_moments = get_invariant_moments(
            test_ground_truth_invariant_moments = get_invariant_block_moments(
                test_dataset_path[dataset][i], start_index_ground_truth, end_index_ground_truth, evaluation_properties=evaluation_properties, n_blocks_invariants=n_blocks_invariants,
            )
            # test_prediction_invariant_moments = get_invariant_moments(
            test_prediction_invariant_moments = get_invariant_block_moments(
                list_output_path[index_plasma_batch], evaluation_properties=evaluation_properties, n_blocks_invariants=n_blocks_invariants,
            )
            index_plasma_batch = index_plasma_batch + 1
            total_time_get_invariant = total_time_get_invariant + time.time() - t1
            print(
                dataset,
                "test_ground_truth_invariant_moments",
                test_ground_truth_invariant_moments,
            )
            print(
                dataset,
                "test_prediction_invariant_moments",
                test_prediction_invariant_moments,
            )
            list_error_invariance.append(
                np.mean(
                    [
                        ((a - b) / np.abs(b))**2
                        for a, b in zip(
                            list(test_prediction_invariant_moments),
                            list(test_ground_truth_invariant_moments),
                        )
                    ]
                )
            )
            print(
                dataset,
                "tf_hw2d dist test groud trutth prediction",
                list_error_invariance[-1],
            )
    total_time_error_metric = time.time() - t0
    print("list_error_invariance:", list_error_invariance)
    print("list_n_files_per_datasets:", list_n_files_per_datasets)
    
    #Each output_file is an individual point for the L2 norm. So if one validation is slightly above the ground truth and one slightly below, they don't cancel out. Both will be squared and both positive.
    evaluation_metrics = {}
    for i, dataset in enumerate(dataset_list):
        start_index = 0
        if i > 0:
            start_index = np.sum(list_n_files_per_datasets[:i]) * n_batches
        end_index = np.sum(list_n_files_per_datasets[: i + 1]) * n_batches
        print("start_index:", start_index)
        print("end_index:", end_index)
        print(
            dataset,
            "dist mean test groud truth prediction",
            np.mean(list_error_invariance[start_index:end_index]),
        )
        evaluation_metrics[dataset] = np.mean(list_error_invariance[start_index:end_index])
    math.use("tensorflow")
    # hw.c1 = save_c1_training    #reset it to training values
    print("total_time_properties_plot:", total_time_properties_plot)
    print("total_time_get_invariant:", total_time_get_invariant)
    print("total_time_error_metric:", total_time_error_metric)
    start_delete_time = time.time()
    if not keep_all_simulation_data:
        for output_path in list_output_path:
            os.remove(output_path)
    total_delete_time = time.time() - start_delete_time
    print("total_delete_time:", total_delete_time)
    return evaluation_metrics

def convert_model_to_tensorrt(dataset_list, valid_batch, save_name_eval, size_input):
    print("Start converting cnn...")
    os.environ["TF_TRT_MAX_ALLOWED_ENGINES"]="40" #Is that why we get illegal instruction errors?
    conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.FP32)
    # input_fn: a generator function that yields input data as a list or tuple,
    # which will be used to execute the converted signature to generate TensorRT
    # engines. Example:
    def my_input_fn():
        # Let's assume a network with 2 input tensors. We generate 3 sets
        # of dummy input data:
        input_shapes = [[(len(dataset_list)*valid_batch,size_input,size_input,1)]]
        #input_shapes = [[(len(dataset_list)*valid_batch,size_input,size_input,9)]]  #JUST FOR THE TEST WITH GRADIENTS AS INPUTS
        for shapes in input_shapes:
            # return a list of input tensors
            yield [np.zeros(x).astype(np.float32) for x in shapes]

    print("Converter...")
    converter = trt.TrtGraphConverterV2(
        #input_saved_model_dir=input_saved_model_dir,
        input_saved_model_dir=save_name_eval,
        conversion_params=conversion_params)

    # requires some data for calibration
    print("Converting...")
    # converter.convert(calibration_input_fn=my_input_fn) #callibration only needed for trt.TrtPrecisionMode.INT8 format??
    converter.convert()
    print("Building...")
    converter.build(input_fn=my_input_fn)
    output_saved_model_dir = save_name_eval+"_CONVERTED"
    print("Saving...")
    converter.save(output_saved_model_dir)
    print("Convertion done.")
    
    # Load converted model and infer
    print("Loading converted model...")
    model_converted = tf.saved_model.load(output_saved_model_dir)
    print('dir(model_converted):', dir(model_converted))
    print('dir(model_converted):', model_converted.signatures.items())
    # func = root.signatures['serving_default']   #What is this line???from the tensorflow example
    func = model_converted.signatures['serving_default']   #What is this line???from the tensorflow example #IT GIVES THE FUNCTION TO USE TO INFER WITH THE MODEL
    print("Converted model inference...")
    output = func(tf.convert_to_tensor(np.ones((len(dataset_list)*valid_batch, size_input, size_input, 1)), dtype=tf.float32))    #TEST OF A PREDICTION
    print("Converted model inference done.")
    return func, converter, model_converted


if __name__ == "__main__":
    print("free GPU mem before fire.Fire(training):", get_gpu_memory())
    print("tf GPU mem:", tf.config.experimental.get_memory_info('GPU:0'))
    print("CPU GPU mem:", [mmm for mmm in map(int, os.popen('free -t -m').readlines()[-1].split()[1:])])
    fire.Fire(training)
