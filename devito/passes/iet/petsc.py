from devito.types.basic import AbstractSymbol, AbstractFunction
from devito.tools import petsc_type_to_ctype, dtype_to_ctype, dtype_to_cstr, dtype_to_petsctype
import numpy as np
from sympy import Expr
from devito.types import LocalObject
from ctypes import POINTER, c_void_p
from devito.ir import Definition



# class PETScObject(AbstractSymbol):
#     @property
#     def _C_ctype(self):
#         ctype = petsc_type_to_ctype(self._dtype)
#         return type(self._dtype, (ctype,), {})
    

class PETScDM(LocalObject):
    dtype = type('DM', (c_void_p,), {})

    
da = PETScDM('da')
defn1 = Definition(da)
print(defn1)




# may need to also inherit from Expr
class PETScFunction(AbstractFunction):
    """
    PETScFunctions.
    """

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        dtype = kwargs.get('dtype')
        if dtype is not None:
            return dtype
        elif grid is not None:
            return grid.dtype
        else:
            return np.float32

    @property
    def dimensions(self):
        return self._dimensions
        
    # @classmethod
    # def __shape_setup__(cls, **kwargs):
    #     grid = kwargs.get('grid')
    #     dimensions = kwargs.get('dimensions')
    #     shape = kwargs.get('shape')

    #     if dimensions is None and shape is None and grid is None:
    #         return None

    #     elif grid is None:
    #         if shape is None:
    #             raise TypeError("Need either `grid` or `shape`")
    #     elif shape is None:
    #         if dimensions is not None and dimensions != grid.dimensions:
    #             raise TypeError("Need `shape` as not all `dimensions` are in `grid`")
    #         shape = grid.shape
    #     elif dimensions is None:
    #         raise TypeError("`dimensions` required if both `grid` and "
    #                         "`shape` are provided")

    #     return shape
        
    @classmethod
    def __indices_setup__(cls, *args, **kwargs):
        grid = kwargs.get('grid')
        dimensions = kwargs.get('dimensions')
        if grid is None:
            if dimensions is None:
                raise TypeError("Need either `grid` or `dimensions`")
        elif dimensions is None:
            dimensions = grid.dimensions

        return tuple(dimensions), tuple(dimensions)

    @property
    def _C_ctype(self):
        petsc_type = dtype_to_petsctype(self.dtype)
        ctype = dtype_to_ctype(self.dtype)
        r = type(petsc_type, (ctype,), {})
        for n in range(len(self.dimensions)):
            r = POINTER(r)
        return r

    @property
    def _C_name(self):
        return self.name
    
    

    
# from devito.ir import Definition
# da = PETScObject('da', dtype='DM')
# tmp = Definition(da)
# print(tmp)

from devito import *
grid = Grid((2, 2))
x, y = grid.dimensions
ptr1 = PETScFunction(name='ptr1', dtype=np.float32, dimensions=grid.dimensions, shape=grid.shape)
defn2 = Definition(ptr1)
# from IPython import embed; embed()
print(str(defn2))



