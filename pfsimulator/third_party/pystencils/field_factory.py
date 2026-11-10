from typing import Tuple

from pystencils import Field, FieldType, DEFAULTS
from pystencils.types import PsType
from pystencils import TypedSymbol


class FieldFactory:
    """Creates 3D pystencils field objects for ParFlow code generation."""

    def __init__(
            self,
            shape: Tuple[TypedSymbol, TypedSymbol, TypedSymbol],
            strides: Tuple[TypedSymbol, TypedSymbol, TypedSymbol],
            dtype: PsType = DEFAULTS.numeric_dtype,
            field_layout: str = "fzyx",
    ):
        self.shape = shape
        self.strides = strides
        self.dtype = dtype
        self.field_layout = field_layout

        self.spatial_dimensions = 3

    def create_new(self, name: str):

        f = Field.create_generic(
            name,
            self.spatial_dimensions,
            index_shape=(),
            dtype=self.dtype,
            field_type=FieldType.GENERIC,
            layout=self.field_layout,
        )

        # overwrite common stride and shape to avoid parameter duplication in kernel
        f.strides = self.strides
        f.shape = self.shape

        return f


