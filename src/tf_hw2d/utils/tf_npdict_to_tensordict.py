# from hw2d.utils.io import *
# from utils_io import *
from phiml import math


def n_o_p_dict(np_dict, batch_indices, unroll_index, batch_size, dx):
    """
    Transforms from dictionnary of numpy arrays to math.Dict of math.Tensor.

    Args:
        np_dict: Dictionnary of numpy arrays. Arrays contain density, omega and phi fields.
        batch_indices: Indices to extract from the fields.
        unroll_index: Index of the unrolled step to extract. 0 is current time step t.
        batch_size: Batch size. (That's just the length of batch_indices)
        dx: Physical distance between grid points, `float` or `Tensor`.
            When passing a vector-valued `Tensor`, the dx values should be listed along `stack_dim`, matching `dims`.

    Returns:
        The math.Dict of the fields as math.Tensor.
    """
    return math.Dict(
        density=math.tensor(
            np_dict["density"][batch_indices, :, :, unroll_index],
            math.batch(batch=batch_size),
            math.spatial(
                y=np_dict["density"].shape[1],
                x=np_dict["density"].shape[2],
            ),
        ),
        omega=math.tensor(
            np_dict["omega"][batch_indices, :, :, unroll_index],
            math.batch(batch=batch_size),
            math.spatial(
                y=np_dict["omega"].shape[1],
                x=np_dict["omega"].shape[2],
            ),
        ),
        phi=math.tensor(
            np_dict["phi"][batch_indices, :, :, unroll_index],
            math.batch(batch=batch_size),
            math.spatial(
                y=np_dict["phi"].shape[1],
                x=np_dict["phi"].shape[2],
            ),
        ),
        age=0,
        dx=dx,
    )


def n_o_p_dict_concat(dict_list, dim="batch"):
    """
    Transforms from list of math dictionnaries to a single math.Dict and concatenates the fields of each dictionnaries.

    Args:
        dict_list: List of dictionnaries to process.
        dim: Dimension to concatenate allong. Defaults to "batch".

    Returns:
        The math.Dict that concatenates the fields from the list of dictionnaries.
    """
    return math.Dict(
        density=math.concat([dict_l.density for dict_l in dict_list], dim=dim),
        omega=math.concat([dict_l.omega for dict_l in dict_list], dim=dim),
        phi=math.concat([dict_l.phi for dict_l in dict_list], dim=dim),
        age=0,
        dx=dict_list[0].dx,
    )
