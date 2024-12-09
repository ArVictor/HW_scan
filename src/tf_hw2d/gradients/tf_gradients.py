"""
PhiML-based Gradient Computation
================================

REQUIRES findiff package!!

This module offers a collection of functions for computing gradients on 2D grids using the PhiML library.
It provides a standard implementation based on array computation suitable for solving the HW2D model, including:

- Basic Gradient Computation (`gradient`): Computes the gradient of a 2D array using central finite differences of order 2 on the borders and order >=2 inside.
- Periodic Gradient (`periodic_gradient`): Computes the gradient with periodic boundary conditions.
- Custom finite differences (`spatial_gradient_highorder`): Computes the gradient with up-/down-wind/central/mixed/mixed_2 scheme, of specified order. "mixed" scheme applies central scheme when possible, otherwise up-/down-wind when not enough ghost cells are available at the borders. "mixed_2" always uses central scheme, but reduces the order to 2 on the borders.
- Finite differences computation (`compute_finite_difference`): Given a list of stencil coefficients, computes the finite difference.
- Shift tensor (`shift_with_pad`): Shifts tensor with padding. This is a workaround for math.shift which doesn't work in our implementation.
- Finite diffrence stencils (`finite_difference_coefficients_shifts`): Returns the coefficients and shifts of a finitte difference stencil given a method and an order.
- Laplace Operations:
    - Iterative Laplace (`periodic_laplace_N`): Computes the Laplace N times successively.
"""

from phiml import math
from typing import Union
import findiff


@math.jit_compile
def periodic_laplace_N(arr: math.Tensor, dx: float, N: int) -> math.Tensor:
    """
    Compute the Laplace of a 2D array using finite differences N times successively with periodic boundary conditions.

    Args:
        a (math.Tensor): Input 2D array.
        dx (float): The spacing between grid points.
        N (int): Number of iterations.

    Returns:
        math.Tensor: The Laplace of the input array with periodic boundary conditions.
    """
    lp_arr = math.laplace(arr, dx=dx, padding="periodic")
    for i in range(1, N):
        lp_arr = math.laplace(lp_arr, dx=dx, padding="periodic")
    return lp_arr


@math.jit_compile
def periodic_gradient(
    input_field: math.Tensor, dx: float, axis: str = "y"
) -> math.Tensor:
    """
    Compute the gradient of a 2D array using finite differences with periodic boundary conditions.

    Args:
        input_field (math.Tensor): Input 2D array.
        dx (float): The spacing between grid points.
        axis (int): Axis along which the gradient is taken

    Returns:
        math.Tensor: Gradient in axis-direction with periodic boundary conditions.
    """
    return math.spatial_gradient(
        input_field, dims=axis, dx=dx, difference="central", padding="periodic"
    )


def gradient(
    input_field: math.Tensor,
    dx: float,
    scheme: str = "mixed_2",
    axis: str = "y",
    order: int = 2,
) -> math.Tensor:
    """
    Compute the gradient of a 2D array using finite differences with periodic boundary conditions. Although periodic, with the "mixed_2" scheme, it only needs one ghost cell on the border.

    Args:
        input_field (math.Tensor): Input 2D array.
        dx (float): The spacing between grid points.
        scheme (str): scheme for the gradient computation
        axis (str): Axis along which the gradient is taken
        order (int): Order of the method for inside cells. Border cells have order 2.

    Returns:
        math.Tensor: Gradient in axis-direction with periodic boundary conditions.
    """
    return spatial_gradient_highorder(
        input_field,
        dims=axis,
        dx=dx,
        scheme=scheme,
        order=order,
        padding="periodic",
    )


def spatial_gradient_highorder(
    grid: math.Tensor,
    dx: Union[float, math.Tensor] = 1,
    scheme: str = "mixed",
    order: int = 2,
    padding: Union[
        math.Extrapolation, float, math.Tensor, str
    ] = math.extrapolation.BOUNDARY,
    dims: math.DimFilter = math.spatial,
    stack_dim: Union[math.Shape, None] = math.channel("gradient"),
    pad=0,
) -> math.Tensor:
    """
    Calculates the spatial_gradient of a scalar channel from finite differences.
    The spatial_gradient vectors are in reverse order, lowest dimension first.

    Args:
        grid: grid values
        dx: Physical distance between grid points, `float` or `Tensor`.
            When passing a vector-valued `Tensor`, the dx values should be listed along `stack_dim`, matching `dims`.
        scheme: Type of difference, one of ('mixed', 'mixed_2', 'forward', 'backward', 'central') (default 'mixed')
        order: (Int, Optional) Accuracy order of the method. Must be even and strictly greater than 1.
        padding: Padding mode.
            Must be one of the following: `Extrapolation`, `Tensor` or number for constant extrapolation, name of extrapolation as `str`.
        dims: (Optional) Dimensions along which the spatial derivative will be computed. sequence of dimension names
        stack_dim: name of the new vector dimension listing the spatial_gradient w.r.t. the various axes
        pad: How many cells to extend the result compared to `grid`.
            This value is added to the internal padding. For non-trivial extrapolations, this gives the correct result while manual padding before or after this operation would not respect the boundary locations.

    Returns:
        `Tensor`: The gradient of grid in dims-dimension.
    """
    grid = math.wrap(grid)
    if stack_dim and stack_dim in grid.shape:
        assert (
            grid.shape.only(stack_dim).size == 1
        ), f"spatial_gradient() cannot list components along {stack_dim.name} because that dimension already exists on grid {grid}"
        grid = grid[{stack_dim.name: 0}]
    dims = grid.shape.only(dims)
    assert (
        dims.rank == 1
    ), f"We only compute one derivative at a time! It makes it easier to deal with backward/forward schemes on the edges."
    dx = math.wrap(dx)
    if dx.vector.exists:
        dx = dx.vector[dims]
        if dx.vector.size in (None, 1):
            dx = dx.vector[0]
    diffs = {}
    # compute all schemes could be optimized? for forward and backward we just need the edges?
    if scheme == "mixed":
        methods = ["central", "forward", "backward"]
    elif scheme == "mixed_2":
        methods = ["central", "forward", "backward"]
    else:
        methods = [scheme]
    for method in methods:
        if scheme != "mixed_all":
            (coefficients_centered, shifts_centered) = get_coefs_and_shifts(
                scheme, method, order
            )
            # if scheme == "mixed_2":
            #     if method != "central":
            #         (
            #             coefficients_centered,
            #             shifts_centered,
            #         ) = finite_difference_coefficients_shifts(2, "central")
            #     else:
            #         (
            #             coefficients_centered,
            #             shifts_centered,
            #         ) = finite_difference_coefficients_shifts(order, "central")
            # else:
            #     (
            #         coefficients_centered,
            #         shifts_centered,
            #     ) = finite_difference_coefficients_shifts(order, method)
            diffs[method] = compute_finite_difference(
                grid,
                coefficients_centered,
                shifts_centered,
                dims,
                padding,
                stack_dim,
                pad,
                dx,
            )
        else:
            diff = math.zeros(dims)
            list_matrix_space_coefs = [math.zeros(dims) for _ in range(order // 2)]
            dict_findiff_central = get_coefs_array(
                order
            )  # coefs only for the positive shifts. Negatives shifts have a flipped sign for the coefs.
            for tmp_order in range(
                order // 2
            ):  # we need for all physical location the coef`ficient of all orders of the central schemes.
                # Create a mask that exclude border elements that cannot be accessed by high order methods.
                mask_range = math.range_tensor(
                    dims
                )  # a mask with simply 0, 1, 2, 3, ...
                mask_range_left = mask_range < tmp_order
                mask_range_right = mask_range >= dims.size - tmp_order
                mask_range_both = math.cast(
                    1 - (mask_range_left + mask_range_right), bool
                )
                # Use the mask the set the matrix spatial coefficients
                for i in range(tmp_order + 1):
                    list_matrix_space_coefs[i] = math.where(
                        mask_range_both,
                        dict_findiff_central[2 * (tmp_order + 1)][i],
                        list_matrix_space_coefs[i],
                    )
            for direction in [
                -1,
                1,
            ]:  # first for the negative shifts (u_{n-4},u_{n-3},u_{n-2},...), then the positive shifts (u_{n+1},u_{n+2},u_{n+3},...)
                shifts = [
                    direction * i for i in range(1, (order // 2) + 1)
                ]  # only positive shifts
                shifted_grids = shift_with_pad(
                    grid, shifts, dims, padding, stack_dim=stack_dim, pad=pad
                )
                # Now we do element-wise multiplications between the spatial coeeficients and their corresponding shifted grids.
                for i in range(order // 2):
                    diff = (
                        diff + direction * list_matrix_space_coefs[i] * shifted_grids[i]
                    )
            diffs[scheme] = (
                diff / dx
            )  # last cells have 2nd order finite difference, and the closer we get to the center the higher the method as more points are available
    if scheme == "mixed" or scheme == "mixed_2":
        mask_range = math.range_tensor(dims)  # a mask with simply 0, 1, 2, 3, ...
        # mask_range_left = mask_range < order // 2
        # mask_range_right = mask_range >= dims.size - order // 2
        diff = math.where(mask_range < order // 2, diffs["forward"], diffs["central"])
        diff = math.where(mask_range >= dims.size - order // 2, diffs["backward"], diff)
    else:
        diff = diffs[scheme]
    return diff


def get_coefs_array(order):
    """
    Returns a dictionnary with only central finite difference coefs for strictly positive shifts of all orders up to input order.

    Args:
        order: (Int) Accuracy order of the method.
    Returns:
        dict: (coefficients, shifts)
    """
    dict_coefs = {}
    for tmp_order in range(2, order + 1, 2):
        dict_coefs[tmp_order] = findiff.coefficients(
            deriv=1, acc=tmp_order, symbolic=False
        )["center"]["coefficients"][(tmp_order // 2) + 1 :]
    return dict_coefs


def get_coefs_and_shifts(
    scheme: str,
    method: str,
    order: int,
) -> tuple:
    """
    Returns coefs and shifts for finite differences according to scheme, method and order.

    Args:
        scheme: Type of difference, one of ('mixed', 'mixed_2', 'forward', 'backward', 'central') (default 'mixed')
        method: List of type of differences.
        order: (Int, Optional) Accuracy order of the method. Must be even and strictly greater than 1.
    Returns:
        tuple: (coefficients, shifts)
    """
    if scheme == "mixed_2":
        if method != "central":
            return finite_difference_coefficients_shifts(2, "central")
        else:
            return finite_difference_coefficients_shifts(order, "central")
    else:
        return finite_difference_coefficients_shifts(order, method)


def compute_finite_difference(
    grid: math.Tensor,
    coefficients: list,
    shifts: list,
    dims: math.DimFilter,
    padding: Union[math.Extrapolation, float, math.Tensor, str],
    stack_dim: Union[math.Shape, None],
    pad: int,
    dx: math.Tensor,
) -> math.Tensor:
    """
    Compute the finite difference.

    Args:
        grid: (math.Tensor) Input tensor to differentiate.
        coefficients: (list) List of coefficients for the finite difference scheme.
        shifts: (list) List of shifts for the finite difference scheme.
        dims: (math.DimFilter) Dimension to differentiate along.
        padding: Padding mode.
            Must be one of the following: `Extrapolation`, `Tensor` or number for constant extrapolation, name of extrapolation as `str`.
        stack_dim: name of the new vector dimension listing the spatial_gradient w.r.t. the various axes
        pad: How many cells to extend the result compared to `grid`.
            This value is added to the internal padding. For non-trivial extrapolations, this gives the correct result while manual padding before or after this operation would not respect the boundary locations.
        dx: (math.Tensor) Size of the grid cells. Actually a "wrap" so native tensor?
    Returns:
        difference (math.Tensor)
    """
    # WHY DOESN'T THAT WORK?????? IS IT THE gradientC=y channel dimension? Or what is it??
    # CAN'T USE math.shift WITH THE NEURAL NETWORK
    # shifted_grids_BEFORE = math.shift(
    #     grid,
    #     shifts,
    #     dims,
    #     padding,
    #     stack_dim=stack_dim,
    #     extend_bounds=pad,
    # )
    # WORKAROUND: shift_with_pad
    shifted_grids = shift_with_pad(
        grid, shifts, dims, padding, stack_dim=stack_dim, pad=pad
    )
    list_coef_mult_grid = [
        coef * shift_g for coef, shift_g in zip(coefficients, shifted_grids)
    ]
    difference = (
        math.sum(
            list_coef_mult_grid,
            dim="0",  # dim='0' to add_up the list and output a shape grid
        )
        / dx
    )
    return difference


def finite_difference_coefficients_shifts(order: int, method: str) -> tuple:
    """
    Returns the coefficients and shifts for a finite difference scheme (central, forward or backward) at a given order.

    Args:
        order: (int) Order of the method.
        method: (str) Name of the method.
    Returns:
        (coefficients (list), shifts (list))
    """
    coefficients = {}
    shifts = {}
    coefficients["central"] = {
        2: (-1 / 2, 1 / 2),
        4: (1 / 12, -2 / 3, 2 / 3, -1 / 12),
        6: (-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60),
        8: (1 / 280, -4 / 105, 1 / 5, -4 / 5, 4 / 5, -1 / 5, 4 / 105, -1 / 280),
        16: (
            1 / 102960,
            -8 / 45045,
            2 / 1287,
            -56 / 6435,
            7 / 198,
            -56 / 495,
            14 / 45,
            -8 / 9,
            8 / 9,
            -14 / 45,
            56 / 495,
            -7 / 198,
            56 / 6435,
            -2 / 1287,
            8 / 45045,
            -1 / 102960,
        ),
    }
    coefficients["forward"] = {
        2: [-3 / 2, 2, -1 / 2],
        4: [-25 / 12, 4, -3, 4 / 3, -1 / 4],
        6: [-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6],
        8: [-761 / 280, 8, -14, 56 / 3, -35 / 2, 56 / 5, -14 / 3, 8 / 7, -1 / 8],
        16: [
            -2436559 / 720720,
            16,
            -60,
            560 / 3,
            -455,
            4368 / 5,
            -4004 / 3,
            11440 / 7,
            -6435 / 4,
            11440 / 9,
            -4004 / 5,
            4368 / 11,
            -455 / 3,
            560 / 13,
            -60 / 7,
            16 / 15,
            -1 / 16,
        ],
    }
    coefficients["backward"] = {
        2: [1 / 2, -2, 3 / 2],
        4: [1 / 4, -4 / 3, 3, -4, 25 / 12],
        6: [1 / 6, -6 / 5, 15 / 4, -20 / 3, 15 / 2, -6, 49 / 20],
        8: [1 / 8, -8 / 7, 14 / 3, -56 / 5, 35 / 2, -56 / 3, 14, -8, 761 / 280],
        16: [
            1 / 16,
            -16 / 15,
            60 / 7,
            -560 / 13,
            455 / 3,
            -4368 / 11,
            4004 / 5,
            -11440 / 9,
            6435 / 4,
            -11440 / 7,
            4004 / 3,
            -4368 / 5,
            455,
            -560 / 3,
            60,
            -16,
            2436559 / 720720,
        ],
    }
    shifts["central"] = {
        2: (-1, 1),
        4: (-2, -1, 1, 2),
        6: (-3, -2, -1, 1, 2, 3),
        8: (-4, -3, -2, -1, 1, 2, 3, 4),
        16: (-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8),
    }
    shifts["forward"] = {
        2: [0, 1, 2],
        4: [0, 1, 2, 3, 4],
        6: [0, 1, 2, 3, 4, 5, 6],
        8: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    }
    shifts["backward"] = {
        2: [-2, -1, 0],
        4: [-4, -3, -2, -1, 0],
        6: [-6, -5, -4, -3, -2, -1, 0],
        8: [-8, -7, -6, -5, -4, -3, -2, -1, 0],
        16: [-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
    }
    if method in coefficients:
        if order in coefficients[method]:
            return coefficients[method][order], shifts[method][order]
        else:
            raise ValueError("Invalid order: {}. Can be 2, 4, 6, or 8.".format(order))
    else:
        raise ValueError(
            "Invalid method: {}. Can be central, forward or backward.".format(method)
        )


def shift_with_pad(grid, shifts, dims, padding, stack_dim, pad):
    """
    Compute the finite difference.

    Args:
        grid: (math.Tensor) Input tensor to differentiate.
        shifts: (list) List of shifts for the finite difference scheme.
        dims: (math.DimFilter) Dimension to differentiate along.
        padding: Padding mode.
            Must be one of the following: `Extrapolation`, `Tensor` or number for constant extrapolation, name of extrapolation as `str`.
        stack_dim: name of the new vector dimension listing the spatial_gradient w.r.t. the various axes
        pad: How many cells to extend the result compared to `grid`.
            This value is added to the internal padding. For non-trivial extrapolations, this gives the correct result while manual padding before or after this operation would not respect the boundary locations.
    Returns:
        List of hifted tensors. (List)
    """
    # mode=math.extrapolation.PERIODIC
    list_shifted_grids = []
    # print('dims:', dims)
    for shift in shifts:
        if shift > 0:
            shift_pair = (0, shift)
            list_shifted_grids.append(
                math.pad(
                    grid, {dims.name: shift_pair}, mode=padding, stack_dim=stack_dim
                ).dimension(dims.name)[shift:]
            )
        elif shift < 0:
            shift_pair = (-shift, 0)
            list_shifted_grids.append(
                math.pad(
                    grid, {dims.name: shift_pair}, mode=padding, stack_dim=stack_dim
                ).dimension(dims.name)[:shift]
            )
        else:
            list_shifted_grids.append(grid)
    return list_shifted_grids
