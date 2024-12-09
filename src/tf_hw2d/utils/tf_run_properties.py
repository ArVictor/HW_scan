import numpy
import h5py
import fire
from tqdm import tqdm
from typing import List
from functools import partial
from tf_hw2d.physical_properties import tf_properties
from phiml import math
import time


def add_data(hf, i, batch_size, name, data, debug, selection):
    if name in selection:
        data = data()
        if "gradient" in data.shape:
            hf[name][i : i + batch_size, ...] = data.numpy(data.shape).reshape(-1)
        else:
            hf[name][i : i + batch_size, ...] = data.numpy(data.shape)
        if debug:
            print(f"{name}: {np.mean(data):.2e}", end="  |  ")


def calculate_properties(
    file_path: str = "",
    batch_size: int = 100,
    property_list: List = [
        "gamma_n",
        "gamma_n_spectral",
        "gamma_c",
        "energy",
        "thermal_energy",
        "kinetic_energy",
        "enstrophy",
        "enstrophy_phi",
    ],
    force_recompute: bool = True,
    is_debug: bool = False,
):
    with h5py.File(file_path, "r+") as h5_file:
        # Parameters
        parameters = dict(h5_file.attrs)
        dx = parameters["dx"]
        c1 = parameters["c1"]
        dt = parameters["dt"]
        steps = len(h5_file["density"])
        # Create Properties in Dataset
        selection = []
        for property_name in property_list:
            if property_name in h5_file.keys():
                print(f"Dataset exists:  {property_name}")
                if force_recompute:
                    del h5_file[property_name]
            if property_name not in h5_file.keys():
                selection.append(property_name)
                h5_file.create_dataset(property_name, (steps,))
                print(f"Created Dataset:  {property_name}")
        # Set up iterator
        iterator = range(0, steps, batch_size)
        if not is_debug:
            iterator = tqdm(iterator, disable=False, mininterval=10)
        # Run Through in Batches
        for i in iterator:
            add = partial(
                add_data,
                hf=h5_file,
                i=i,
                batch_size=batch_size,
                debug=is_debug,
                selection=selection,
            )
            n = h5_file["density"][i : i + batch_size]
            p = h5_file["phi"][i : i + batch_size]
            o = h5_file["omega"][i : i + batch_size]            
            n = math.tensor(
                n, math.batch(b=n.shape[0]), math.spatial(y=n.shape[1], x=n.shape[2])
            )
            p = math.tensor(
                p, math.batch(b=p.shape[0]), math.spatial(y=p.shape[1], x=p.shape[2])
            )
            o = math.tensor(
                o, math.batch(b=o.shape[0]), math.spatial(y=o.shape[1], x=o.shape[2])
            )
            if is_debug:
                print(
                    f"Timeframe ({i*dt}-{(i + batch_size)*dt})",
                    end="\n  ",
                )
            # Add properties if they were selected
            add(
                name="gamma_n", data=partial(tf_properties.get_gamma_n, n=n, p=p, dx=dx)
            )
            add(
                name="gamma_n_spectral",
                data=partial(tf_properties.get_gamma_n_spectrally, n=n, p=p, dx=dx),
            )
            add(
                name="gamma_c",
                data=partial(tf_properties.get_gamma_c, n=n, p=p, c1=c1, dx=dx),
            )
            add(
                name="energy", data=partial(tf_properties.get_energy, n=n, phi=p, dx=dx)
            )
            add(
                name="thermal_energy",
                # data=partial(tf_properties.get_energy_N_spectrally, n=n),
                data=partial(tf_properties.get_thermal_energy, n=n),
            )
            add(
                name="kinetic_energy",
                # data=partial(tf_properties.get_energy_V_spectrally, p=p, dx=dx),
                data=partial(tf_properties.get_kinetic_energy_victor, phi=p, dx=dx),
            )
            add(
                name="enstrophy",
                data=partial(tf_properties.get_enstrophy, n=n, omega=o, dx=dx),
            )
            add(
                name="enstrophy_phi",
                data=partial(tf_properties.get_enstrophy_phi, n=n, phi=p, dx=dx),
            )
            if is_debug:
                print()
    return


if __name__ == "__main__":
    fire.Fire(calculate_properties)
