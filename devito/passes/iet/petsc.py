from devito.types.basic import AbstractSymbol, AbstractFunction
from devito.tools import petsc_type_to_ctype, dtype_to_ctype, dtype_to_cstr
import numpy as np
from sympy import Expr
from ctypes import POINTER



class PETScObject(AbstractSymbol):
    """
    PETScObjects.
    """
    @property
    def _C_ctype(self):
        ctype = petsc_type_to_ctype(self._dtype)
        return type(self._dtype, (ctype,), {})
    


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
        """Tuple of Dimensions representing the object indices."""
        return self._dimensions

    @classmethod
    def __indices_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        shape = kwargs.get('shape')
        dimensions = kwargs.get('dimensions')

        if dimensions is None and shape is None and grid is None:
            return (), ()

        elif grid is None:
            if dimensions is None:
                raise TypeError("Need either `grid` or `dimensions`")
        elif dimensions is None:
            dimensions = grid.dimensions

        return dimensions, dimensions
        
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
        
    # @classmethod
    # def __indices_setup__(cls, *args, **kwargs):
    #     grid = kwargs.get('grid')
    #     dimensions = kwargs.get('dimensions')
    #     if grid is None:
    #         if dimensions is None:
    #             raise TypeError("Need either `grid` or `dimensions`")
    #     elif dimensions is None:
    #         dimensions = grid.dimensions

    #     return tuple(dimensions), tuple(dimensions)

    @property
    def _C_ctype(self):
        # from IPython import embed; embed()
        ctypename = 'Petsc%s' % dtype_to_cstr(self._dtype).capitalize()
        ctype = dtype_to_ctype(self.dtype)
        r = POINTER(type(ctypename, (ctype,), {}))
        for n in range(len(self.dimensions)-1):
            r = POINTER(r)
        return r
    
    # @property
    # def _C_ctype(self):
    #     # from IPython import embed; embed()
    #     ctypename = 'Petsc%s' % dtype_to_cstr(self._dtype).capitalize()
    #     ctype = dtype_to_ctype(self.dtype)
    #     r = type(ctypename, (ctype,), {})
    #     # for n in range(len(self.dimensions)-1):
    #     #     r = POINTER(r)
    #     from IPython import embed; embed()
    #     return r

    @property
    def _C_name(self):
        return self.name
    
    

    
from devito.ir import Definition
da = PETScObject('da', dtype='DM')
tmp = Definition(da)
# print(tmp)

from devito import *
grid = Grid((2, 2))
x, y = grid.dimensions
# pointer and const functionality
ptr1 = PETScFunction(name='ptr1', dtype=np.int32, dimensions=grid.dimensions, shape=grid.shape, is_const=True)
defn1 = Definition(ptr1)
from IPython import embed; embed()
print(str(defn1))



