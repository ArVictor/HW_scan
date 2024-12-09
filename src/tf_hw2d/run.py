import fire
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Iterable

from tf_hw2d.model import HW

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

from phiml import math


def run(
    step_size: float = 0.025,
    end_time: float = 1_000,
    grid_pts: int = 512,
    k0: float = 0.15,
    N: int = 3,
    nu: float = 5.0e-08,
    c1: float = 1.0,
    kappa_coeff: float = 1.0,
    arakawa_coeff: float = 1.0,
    poisson_method: str = "fourier",
    seed: int or None = None,
    init_type: str = "normal",
    init_scale: float = 1 / 100,
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
        "gamma_c",
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
):
    """
    Run the simulation with the given parameters.

    Args:
        step_size (float, optional): Incremental step for simulation progression. Defaults to 0.025.
        end_time (float, optional): Duration till the simulation should run. Defaults to 1_000.
        grid_pts (int, optional): Grid points. Suggested: 128 for coarse, 1024 for fine. Defaults to 512.
        k0 (float, optional): Determines k-focus. Suggested: 0.15 for high-k, 0.0375 for low-k. Defaults to 0.15.
        N (int, optional): Dissipation exponent's half value. Defaults to 3.
        nu (float, optional): Viscosity. Suggested: 5e-10 for coarse-large, 1e-4 for fine-small. Defaults to 5.0e-08.
        c1 (float, optional): Transition scale between hydrodynamic and adiabatic. Suggested values: 0.1, 1, 5. Defaults to 1.0.
        kappa_coeff (float, optional): Coefficient of d/dy phi. Defaults to 1.0.
        arakawa_coeff (float, optional): Coefficient of Poisson bracket [A,B] implemented with Arakawa Scheme. Defaults to 1.0.
        poisson_method (str, optional): Method to solve the Poisson equation. Either "fourier" or "CG".
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

    Returns:
        None: The function saves simulation data or generates a movie as specified.
    """
    # math.use('numpy')
    math.use("tensorflow")
    # Unpacking
    y = grid_pts
    x = grid_pts
    L = 2 * math.PI / k0  # Physical box size
    dx = L / x  # Grid resolution
    steps = int(end_time / step_size)  # Number of Steps until end_time
    snap_count = steps // snaps + 1  # number of snapshots
    field_list = ("density", "omega", "phi")
    math.seed(seed)

    dim_order = ("y", "x", "gradient")

    def get_random_field(x, y):
        return math.random_normal(math.spatial(y=y, x=x))

    # Physics
    physics_params = dict(
        dx=dx,
        N=N,
        c1=c1,
        nu=nu,
        k0=k0,
        arakawa_coeff=arakawa_coeff,
        kappa_coeff=kappa_coeff,
        poisson_method=poisson_method,
    )
    # Initialize Plasma

    plasma = math.Dict(
        density=get_random_field(x, y) * init_scale,
        omega=get_random_field(x, y) * init_scale,
        phi=get_random_field(x, y) * init_scale,
        age=0,
        dx=dx,
    )

    # File Handling
    if output_path:
        buffer = {
            field: np.zeros((buffer_size, y, x), dtype=np.float32)
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
            save_params = get_save_params(physics_params, step_size, snaps, x, y)
            create_appendable_h5(
                output_path,
                save_params,
                chunk_size=100,
            )
            output_params["buffer_index"] = save_to_buffered_h5(
                new_val=plasma,
                buffer_size=buffer_size,
                dim_order=dim_order,
                **output_params,
            )

    # Setup Simulation
    hw = HW(**physics_params, debug=debug)
    plasma["phi"] = hw.get_phi(
        plasma.omega, physics_params["dx"], x0=0.0 * plasma["omega"]
    )

    # Run Simulation
    print("Running simulation...")
    for i in tqdm(range(1, steps + 1)):
        # Progress one step, alternatively: hw.euler_step()
        plasma = hw.rk4_step(plasma, dt=step_size, dx=dx)
        # plasma = hw.euler_step(plasma, dt=step_size, dx=dx)

        # Save to records
        if output_path and i % snaps == 0:
            output_params["buffer_index"] = save_to_buffered_h5(
                new_val=plasma,
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
    if output_path and output_params["buffer_index"] > 0:
        append_h5(**output_params)

    # Get Performance stats
    hw.print_log()

    if properties and output_path:
        print(f"Calculating properties...")
        calculate_properties(
            file_path=output_path,
            batch_size=buffer_size,
            property_list=properties,
            force_recompute=True,
            is_debug=False,
        )

    if plot_properties and output_path:
        print(f"Plotting properties...")
        plot_timetraces(
            file_path=output_path,
            out_path=None,
            properties=plot_properties,
            t0=0,
            t0_std=300,
        )

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


if __name__ == "__main__":
    fire.Fire(run)
