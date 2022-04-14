# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""
Perform transformations on Dimensions
=====================================

Provide functions and classes to build a Space which an algorithm can operate on.

"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.dummy import Array
import operator
from typing import Callable, ClassVar, List, Sequence, TypeVar, Union, Generic, cast
import copy
import functools
import itertools
from abc import ABC, abstractmethod
from typing_extensions import Protocol
import numpy

from orion.algo.space import Categorical, Dimension, Fidelity, Integer, Real, Space
from orion.core.utils import format_trials
from orion.core.utils.flatten import flatten

NON_LINEAR = ["loguniform", "reciprocal"]


T_in = TypeVar("T_in")
T_out = TypeVar("T_out")
D_in = TypeVar("D_in", bound=Dimension)
D_out = TypeVar("D_out", bound=Dimension)

# pylint: disable=unused-argument
@functools.singledispatch
def build_transform(
    dim: Dimension, type_requirement: str | None, dist_requirement: str | None
):
    """Base transformation factory

    Parameters
    ----------
    dim: `orion.algo.space.Dimension`
        A dimension object which may need transformations to match provided requirements.
    type_requirement: str, None
        String defining the requirement of the algorithm. It can be one of the following
        - 'real', the dim should be transformed so type is `orion.algo.space.Real`
        - 'integer', the dim should be transformed so type is `orion.algo.space.Integer`
        - 'numerical', the dim should be transformed so type is either `orion.algo.space.Integer` or
        `orion.algo.space.Real`
        - None, no requirement
    dist_requirement: str, None
        String defining the distribution requirement of the algorithm.
        - 'linear', any dimension with logarithmic prior while be linearized
        - None, no requirement

    """
    return []


_transforms_mapping: dict[
    tuple[type[Dimension], type[Dimension]], Callable[[Dimension], Dimension]
] = {
    (Categorical, Real): lambda dim: OneHotEncode(Enumerate(dim)),
    (Categorical, Integer): Enumerate,
}


def register_transform(
    in_type: type[D_in], out_type: type[D_out]
) -> Callable[[Callable[[D_in], D_out]], Callable[[D_in], D_out]]:
    """Register a transformation function for a given input and output type"""

    def decorator(func: Callable[[D_in], D_out]) -> Callable[[D_in], D_out]:
        """Decorator for registering a transformation function"""
        _transforms_mapping[in_type, out_type] = func
        return func

    return decorator


import inspect


def _(dim: Dimension, type_requirement: dype[Dimension], dist_requirement: str):
    transformers = []
    if type_requirement == "real":
        transformers.extend(
            [Enumerate(dim.categories), OneHotEncode(len(dim.categories))]
        )
    elif type_requirement in ["integer", "numerical"]:
        transformers.append(Enumerate(dim.categories))

    return transformers


@build_transform.register(Fidelity)
def _(dim, type_requirement, dist_requirement):
    return []


@build_transform.register(Integer)
def _(dim, type_requirement, dist_requirement):
    transformers = []
    if dist_requirement == "linear" and dim.prior_name[4:] in NON_LINEAR:
        transformers.extend([Reverse(Quantize()), Linearize()])
        # NOTE: we do not turn back to integer even though linearize outputs real
        #       otherwise the mapping from exp(int) to int squashes out lots of possible values.
    elif type_requirement == "real":
        transformers.append(Reverse(Quantize()))

    return transformers


@build_transform.register(Real)
def _(dim, type_requirement, dist_requirement):
    transformers = []
    if dim.precision is not None:
        transformers.append(Precision(dim.precision))

    if dist_requirement == "linear" and dim.prior_name in NON_LINEAR:
        transformers.append(Linearize())
    elif type_requirement == "integer":
        # NOTE: This may cause out-of-bound errors for rounded reals. Not fixed for now
        #       because there are no foreseeable algorithms that may require integer type.
        transformers.append(Quantize())

    return transformers


def transform(original_space, type_requirement, dist_requirement):
    """Build a transformed space"""
    space = TransformedSpace(original_space)
    for dim in original_space.values():
        transformers = build_transform(dim, type_requirement, dist_requirement)
        space.register(
            TransformedDimension(
                transformer=Compose(transformers, dim.type), original_dimension=dim
            )
        )
    return space


def reshape(space, shape_requirement):
    """Build a reshaped space"""
    if shape_requirement is None:
        return space

    # We assume shape_requirement == 'flattened'

    reshaped_space = ReshapedSpace(space)

    for dim_index, dim in enumerate(space.values()):
        if not dim.shape:
            reshaped_space.register(
                ReshapedDimension(
                    transformer=Identity(dim.type),
                    original_dimension=dim,
                    index=dim_index,
                )
            )
        else:
            for index in itertools.product(*map(range, dim.shape)):
                key = f'{dim.name}[{",".join(map(str, index))}]'
                reshaped_space.register(
                    ReshapedDimension(
                        transformer=View(dim.shape, index, dim.type),
                        original_dimension=dim,
                        name=key,
                        index=dim_index,
                    )
                )

    return reshaped_space


def build_required_space(
    original_space, type_requirement=None, shape_requirement=None, dist_requirement=None
):
    """Build a :class:`orion.algo.space.Space` object which agrees to the `requirements` imposed
    by the desired optimization algorithm.

    It uses appropriate cascade of `Transformer` objects per `orion.algo.space.Dimension`
    contained in `original_space`. `ReshapedTransformer` objects are used above
    the `Transformer` if the optimizatios algorithm requires flattened dimensions.

    Parameters
    ----------
    original_space : `orion.algo.space.Space`
        Original problem's definition of parameter space given by the user to Oríon.
    type_requirement: str, None
        String defining the requirement of the algorithm. It can be one of the following
        - 'real', the dim should be transformed so type is `orion.algo.space.Real`
        - 'integer', the dim should be transformed so type is `orion.algo.space.Integer`
        - 'numerical', the dim should be transformed so type is either `orion.algo.space.Integer` or
        `orion.algo.space.Real`
        - None, no requirement
    shape_requirement: str, None
        String defining the shape requirement of the algorithm.
        - 'flattened', any dimension with shape > 1 will be flattened
        - None, no requirement
    dist_requirement: str, None
        String defining the distribution requirement of the algorithm.
        - 'linear', any dimension with logarithmic prior while be linearized
        - None, no requirement
    """
    space = transform(original_space, type_requirement, dist_requirement)
    space = reshape(space, shape_requirement)

    return space


class Transform(ABC, Generic[T_in, T_out]):
    def __call__(self, input: ArrayLike[T_in]) -> ArrayLike[T_out]:
        """Transform a point from domain dimension to the target dimension."""
        return self.forward(input)

    @abstractmethod
    def forward(self, input: ArrayLike[T_in]) -> ArrayLike[T_out]:
        """Transform a point from domain dimension to the target dimension."""

    @abstractmethod
    def reverse(self, output: ArrayLike[T_out]) -> ArrayLike[T_in]:
        """Reverse transform a point from target dimension to the domain dimension."""


class Transformer(Protocol[D_in, D_out]):
    """Define an (injective) function and its inverse. Base transformation class.

    Attributes
    ----------
    target_type: str
        Defines the type of the target space of the forward function.
        It can provide one of the values: ``['real', 'integer', 'categorical']``.
    domain_type: str
        Is similar to ``target_type`` but it refers to the domain.
        If it is ``None``, then it can receive inputs of any type.
    """

    def __call__(self, dim: D_in) -> D_out:
        """Transform a dimension.

        Parameters
        ----------
        dim: `orion.algo.space.Dimension`
            Dimension to be transformed.

        Returns
        -------
        `orion.algo.space.Dimension`
            Transformed dimension.

        """
        return dim.transform(self)


def identity(dim: Dimension) -> Dimension:
    """Identity transformer.

    Parameters
    ----------
    dim: `orion.algo.space.Dimension`
        Dimension to be transformed.

    Returns
    -------
    `orion.algo.space.Dimension`
        Transformed dimension.

    """
    return dim


class Compose(List[Transform], Transform[T_in, T_out]):
    """Initialize composite transformer with a list of `Transformer` objects
    and domain type on which it will be applied.
    """

    def __call__(self, input: T_in) -> T_out:
        """Apply transformers in the increasing order of the `transformers` list."""
        value = input
        for func in self:
            value = func(value)  # type: ignore
        out = cast(T_out, value)
        return out

    def reverse(self, output: T_out) -> T_in:
        """Reverse transformation by reversing in the opposite order of the `transformers` list."""
        value = output
        for transform in reversed(self):
            value = transform.reverse(value)  # type: ignore
        result = cast(T_in, value)
        return result

    # @property
    # def domain_type(self):
    #     """Return base domain type."""
    #     return self[0].domain_type

    # @property
    # def target_type(self):
    #     """Infer type of the tranformation target."""
    #     return self[-1].target_type


@dataclass
class Precision(Transformer[Real, Real]):
    """Round real numbers to requested precision."""

    precision: int = 4

    def forward(self, point: ArrayLike[float]) -> ArrayLike[float]:
        """Round `point` to the requested precision, as numpy arrays."""
        # numpy.format_float_scientific precision starts at 0
        if isinstance(point, (list, tuple)) or (
            isinstance(point, numpy.ndarray) and point.shape
        ):
            format_float = numpy.vectorize(
                lambda x: numpy.format_float_scientific(x, precision=self.precision - 1)
            )
            point = format_float(point)
            to_float = numpy.vectorize(float)
            point = to_float(point)
        else:
            point = float(
                numpy.format_float_scientific(point, precision=self.precision - 1)
            )

        return numpy.asarray(point)

    def reverse(self, transformed_point: ArrayLike[float]) -> ArrayLike[float]:
        """There isn't really a reverse in this case. We can't add more decimals out of nowhere).
        We just return the same values.
        """
        return transformed_point


T = TypeVar("T")
ArrayLike = Union[T, numpy.ndarray, Sequence[T]]


def _quantize(point: ArrayLike[float]) -> ArrayLike[int]:
    """Round `point` and then cast to integers, as numpy arrays."""
    quantized = numpy.round(numpy.asarray(point)).astype(int)

    if numpy.any(numpy.isinf(point)):
        isinf = int(numpy.isinf(point))
        quantized = (
            isinf * (quantized - 1) * int(numpy.sign(point))
            + (1 - isinf) * (quantized - 1)
        ).astype(int)

    return quantized


def _reverse_quantize(transformed_point: ArrayLike[int]) -> ArrayLike[float]:
    """Cast `transformed_point` to floats, as numpy arrays."""
    return numpy.asarray(transformed_point).astype(float)


def Quantize(dim: Real) -> Integer:
    """Transform real numbers to integers. Isn't perfectly reversible."""
    # todo:
    lower = _quantize(dim.lower)
    upper = _quantize(dim.upper)
    out = Integer(name=dim.name, prior=dim.prior, lower=lower, upper=upper)
    out.transform = _quantize
    out.reverse = _reverse_quantize
    return out


def Enumerate(dim: Categorical) -> Integer:
    """Enumerate categories.

    Effectively transform from a list of objects to a range of integers.
    """

    map_dict = {cat: i for i, cat in enumerate(dim.categories)}
    _map = numpy.vectorize(lambda x: dim.categories[x], otypes=[object])
    _imap = numpy.vectorize(lambda x: map_dict[x], otypes="i")

    # todo:
    out = Integer(name=dim.name, low=0, high=len(dim.categories) - 1, prior=dim.prior)
    out.transform = _map
    out.reverse = _imap
    return out


@dataclass
class OneHotEncode(Transformer[Integer, Real]):
    """Encode categories to a 1-hot integer space representation."""

    num_cats: int

    domain_type: ClassVar[type[Integer]] = Integer
    target_type: ClassVar[type[Real]] = Real

    def __call__(self, dim: Integer) -> Real:
        out = Real(
            name=dim.name, prior=dim.prior
        )  # TODO: Pass all the right arguments.
        out.transformer = self
        return out

    def forward(self, point: ArrayLike[int]) -> ArrayLike[float]:
        """Match a `point` containing integers to real vector representations of them.

        If the upper bound of integers supported by an instance of `OneHotEncode`
        is less or equal to 2, then cast them to floats.

        .. note:: This transformation possibly appends one more tensor dimension to `point`.
        """
        point_ = numpy.asarray(point)
        assert (
            numpy.all(point_ < self.num_cats)
            and numpy.all(point_ >= 0)
            and numpy.all(point_ % 1 == 0)
        )

        if self.num_cats <= 2:
            return numpy.asarray(point_, dtype=float)

        hot = numpy.zeros(self.infer_target_shape(point_.shape))
        grid = numpy.meshgrid(
            *[numpy.arange(dim) for dim in point_.shape], indexing="ij"
        )
        hot[grid + [point_]] = 1
        return hot

    def reverse(self, transformed_point: ArrayLike[float]) -> ArrayLike[int]:
        """Match real vector representations to integers using an argmax function.

        If the number of dimensions is exactly 2, then use 0.5 as a decision boundary,
        and convert representation to integers 0 or 1.

        If the number of dimensions is exactly 1, then return zeros.

        .. note:: This reverse transformation possibly removes the last tensor dimension
           from `transformed_point`.
        """
        point_ = numpy.asarray(transformed_point)
        if self.num_cats == 2:
            return (point_ > 0.5).astype(int)
        elif self.num_cats == 1:
            return numpy.zeros_like(point_, dtype=int)

        assert point_.shape[-1] == self.num_cats
        return point_.argmax(axis=-1)


class Linearize(Transformer[Real, Real]):
    """Transform real numbers from loguniform to linear."""

    domain_type: ClassVar[type[Dimension]] = Real
    target_type: ClassVar[type[Dimension]] = Real

    def __call__(self, dim: Real) -> Real:
        lower = self.forward(dim.lower)
        upper = self.forward(dim.upper)
        out = Real(name=dim.name, lower=lower, upper=upper, prior="uniform")
        out.transformer = self

    def forward(self, point: ArrayLike[float]) -> ArrayLike[float]:
        """Linearize logarithmic distribution."""
        return numpy.log(numpy.asarray(point))

    def reverse(self, transformed_point: ArrayLike[float]) -> ArrayLike[float]:
        """Turn linear distribution to logarithmic distribution."""
        return numpy.exp(numpy.asarray(transformed_point))


ArrayDim = TypeVar("ArrayDim", Integer, Real)


@dataclass
class View(Transformer[ArrayDim, Union[Real, Integer]]):
    """Look-up single index in a dimensions with shape > 1"""

    index: int

    def __call__(self):
        raise dimensions

    def forward(self, point: ArrayLike[T]) -> T:
        """Only return one element of the group"""
        return numpy.array(point)[self.index]

    def reverse(self, transformed_point, index=None):
        """Only return packend point if view of first element, otherwise drop."""
        subset = transformed_point[index : index + numpy.prod(self.shape)]
        return numpy.array(subset).reshape(self.shape)


@dataclass(frozen=True)
class Reshape(Transformer[ArrayDim, ArrayDim]):
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...] = (1,)

    def __call__(self, dim: ArrayDim) -> ArrayDim:
        # TODO:
        lower = self.forward(dim.lower)
        upper = self.forward(dim.upper)
        reshaped_dim = type(dim)(lower=lower, upper=upper, shape=upper.shape)
        return reshaped_dim

    def forward(self, point: ArrayLike) -> ArrayLike:
        return numpy.array(point).reshape(self.output_shape)

    def reverse(self, point: ArrayLike) -> ArrayLike:
        return numpy.array(point).reshape(self.input_shape)


TransformedSpace = Space


def change_trial_params(trial, point, space):
    """Convert params in Param objects and update trial"""
    new_trial = copy.copy(trial)
    # pylint: disable=protected-access
    new_trial._params = format_trials.tuple_to_trial(point, space)._params
    return new_trial
