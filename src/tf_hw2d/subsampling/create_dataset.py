import fire
import h5py
from tqdm import tqdm
from typing import Iterable

from tf_hw2d.utils.tf_io import (
    create_appendable_h5,
    save_to_buffered_h5,
    append_h5,
    continue_h5_file,
)

from tf_hw2d.utils.plot.movie import create_movie

from tf_hw2d.utils.tf_run_properties import calculate_properties, add_data

from tf_hw2d.utils.plot.timetrace import plot_timetraces
from tf_hw2d.subsampling import subsampler

from phiml import math

from functools import partial


def create_dataset(
    original_path: str = "",
    new_path: str = "",
    sampling_method: str = "fourier",
    use_old_properties: bool = True,
    spatial_downsampling_ratio: float = 4.0,
    time_downsampling_ratio: int = 1,
    force_recompute: bool = True,
    buffer_size: int = 100,
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
    Create a subsampled dataset from a simulation.

    Args:
        original_path (str): Path to the original simulation data.
        new_path (str): Where to save the new data.
        sampling_method (str): Method for spatial sampling. Can be 'linear', 'fourier' or 'wavelet'.
        use_old_properties (bool): If True, physical properties of the new data set are copied from the original data. Else properties are recomputed on the new dataset. Defaults to True. ONLY TESTED AS True, NOT WITH False! TODO!
        spatial_downsampling_ratio (float, optional): Ratio of original size over subsampled size in space.
        time_downsampling_ratio (float, optional): Ratio of original size over subsampled size in time.
        force_recompute (bool): If True, overwrites previous existing fields in the new dataset.
        buffer_size (int, optional): Size of buffer for storage. Defaults to 100.
        original_path (str, optional): Where to save the simulation data. Defaults to ''.
        movie (bool, optional): If True, generate a movie out of the new dataset. Defaults to True.
        min_fps (int, optional): Parameter for movie generation. Defaults to 10.
        dpi (int, optional): Parameter for movie generation. Defaults to 75.
        speed (int, optional): Parameter for movie generation. Defaults to 5.
        debug (bool, optional): Enable or disable debug mode. Defaults to False.
        properties (Iterable[str], optional): List of properties to calculate for the saved file.
        plot_properties (Iterable[str], optional): List of properties to plot a timetrace for for the new file.

    Returns:
        None: The function saves a new dataset or generates a movie as specified.
    """
    # math.use('numpy')
    math.use("tensorflow")
    # Unpacking

    if original_path == new_path:
        print(
            f"ERROR create_dataset.py create_dataset() - original_path and new_path are identical:{original_path} == {new_path}"
        )
        return
    if sampling_method == "fourier":
        spatial_sampling_function = subsampler.fourier
    elif sampling_method == "linear":
        spatial_sampling_function = subsampler.linear
    elif sampling_method == "wavelet":
        spatial_sampling_function = subsampler.wavelet
    with h5py.File(original_path, "r") as h5_file_original:
        # Parameters
        parameters = dict(h5_file_original.attrs)
        dx = parameters["dx"]
        dt = parameters["dt"]
        original_y = parameters["y"]
        original_x = parameters["x"]
        steps = len(h5_file_original["density"])
        # New Parameters
        new_dt = parameters["dt"] * time_downsampling_ratio
        new_frame_dt = parameters["frame_dt"] * time_downsampling_ratio
        new_y = int(original_y // spatial_downsampling_ratio)
        new_x = int(original_x // spatial_downsampling_ratio)
        new_steps = int((steps - 1) // time_downsampling_ratio) + 1
        # Buffer size needs to be a multiple of time_downsampling_ratio
        buffer_size = int(
            time_downsampling_ratio * (buffer_size // time_downsampling_ratio)
        )
        # Set up iterator original
        iterator = range(0, steps, buffer_size)
        if not debug:
            iterator = tqdm(iterator)  # Create
        # Create new dataset file
        parameters["y"] = new_y
        parameters["x"] = new_x
        parameters["grid_pts"] = new_x
        L = 2 * math.PI / parameters["k0"]  # Physical box size
        new_dx = L / new_x  # Grid resolution
        parameters["dx"] = new_dx
        parameters["dt"] = new_dt
        parameters["frame_dt"] = new_frame_dt
        create_appendable_h5(
            new_path,
            parameters,
            chunk_size=buffer_size // time_downsampling_ratio,
        )
        with h5py.File(new_path, "r+") as h5_file_new:
            # Create Properties in new Dataset
            selection = []
            for property_name in h5_file_original.keys():
                if property_name in h5_file_new.keys():
                    print(f"Dataset exists:  {property_name}")
                    if force_recompute:
                        del h5_file_new[property_name]
                if property_name not in h5_file_new.keys():
                    selection.append(property_name)
                    if h5_file_original[property_name].ndim == 1:
                        shappe_new_data = (new_steps,)
                    elif h5_file_original[property_name].ndim == 3:
                        shappe_new_data = (
                            new_steps,
                            new_y,
                            new_x,
                        )  # IS THAT THE RIGHT ORDER Y,X??
                    else:
                        print(
                            f"ERROR create_dataset.py create_dataset()    h5_file_new.create_dataset - Unknown number of dimensions: tmp_array.ndim"
                        )
                    h5_file_new.create_dataset(property_name, shappe_new_data)
                    print(f"Created Dataset:  {property_name}")
            # Run Through in Batches
            original_data_time_subsampled = (
                {}
            )  # is it faster to have this line here or in the loop lower?
            new_data = {}  # is it faster to have this line here or in the loop lower?
            for i in iterator:
                add = partial(
                    add_data,
                    hf=h5_file_new,
                    i=i // time_downsampling_ratio,
                    batch_size=buffer_size // time_downsampling_ratio,
                    debug=debug,
                    selection=selection,
                )
                if debug:
                    print(
                        f"Timeframe ({i*dt}-{(i + buffer_size)*dt})",
                        end="\n  ",
                    )
                # get original fields with time subsampling
                for field in selection:
                    # tmp_array = h5_file_original[field][i : i + buffer_size, ...]
                    tmp_array = h5_file_original[field][
                        i : i + buffer_size : time_downsampling_ratio, ...
                    ]
                    if tmp_array.ndim == 1:
                        original_data_time_subsampled[field] = math.tensor(
                            tmp_array,
                            math.batch(b=tmp_array.shape[0]),
                        )
                    else:
                        if tmp_array.ndim == 2:
                            spatial_dim = math.spatial(x=tmp_array.shape[1])
                        elif tmp_array.ndim == 3:
                            spatial_dim = math.spatial(
                                y=tmp_array.shape[1], x=tmp_array.shape[2]
                            )
                        else:
                            print(
                                f"ERROR create_dataset.py create_dataset() - Unknown number of dimensions: tmp_array.ndim"
                            )
                        original_data_time_subsampled[field] = math.tensor(
                            tmp_array,
                            math.batch(b=tmp_array.shape[0]),
                            spatial_dim,
                        )
                # Compute spatially subsampled frames
                for field in selection:
                    if original_data_time_subsampled[field].rank == 1:  # 0D properties
                        if use_old_properties:
                            # compute new properties from downsampled data
                            new_data[field] = original_data_time_subsampled[field]
                    elif original_data_time_subsampled[field].rank == 2:
                        # what property is 1D?
                        pass
                    elif original_data_time_subsampled[field].rank == 3:
                        new_data[field] = spatial_sampling_function(
                            original_data_time_subsampled[field],
                            original_x,
                            original_y,
                            new_x,
                            new_y,
                        )
                        print(
                            "DEBUG:",
                            "mean original",
                            math.mean(original_data_time_subsampled[field]),
                            "std original",
                            math.std(original_data_time_subsampled[field]),
                        )
                        print(
                            "DEBUG:",
                            "mean newnewne",
                            math.mean(new_data[field]),
                            "std newnewne",
                            math.std(new_data[field]),
                        )
                for field in selection:
                    if original_data_time_subsampled[field].rank == 1:  # 0D properties
                        if not use_old_properties:
                            # compute new properties from downsampled data
                            # new_data[field] = CALCULATE PROPERTIES FOR DATA IN NEW_DATA
                            pass
                # Write new frames to new file
                for field in selection:
                    add(
                        name=field,
                        data=lambda: new_data[field],
                    )

                if debug:
                    print()
    field_list = ("density", "omega", "phi")

    if not use_old_properties:
        if properties and new_path:
            print(f"Calculating properties...")
            calculate_properties(
                file_path=new_path,
                batch_size=buffer_size,
                property_list=properties,
                force_recompute=True,
                is_debug=False,
            )

    if plot_properties and new_path:
        print(f"Plotting properties...")
        plot_timetraces(
            file_path=new_path,
            out_path=None,
            properties=plot_properties,
            t0=0,
            t0_std=300,
        )
    if movie and new_path:
        print(f"Generating movie...")
        create_movie(
            input_filename=new_path,
            output_filename=new_path,
            t0=0,
            t1=None,
            plot_order=field_list,
            min_fps=min_fps,
            dpi=dpi,
            speed=speed,
        )
    return


if __name__ == "__main__":
    fire.Fire(create_dataset)
