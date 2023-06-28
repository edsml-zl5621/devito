from ctypes import byref
from ctypes import POINTER

import sympy

from devito.tools import Pickable, as_tuple, sympy_mutex
from devito.types.args import ArgProvider
from devito.types.caching import Uncached
from devito.types.basic import Basic
from devito.types.utils import CtypesFactory, DimensionTuple
from devito.tools import dtype_to_cstr, dtype_to_ctype


__all__ = ['Object', 'LocalObject', 'CompositeObject', 'PetscObject']


class AbstractObject(Basic, sympy.Basic, Pickable):

    """
    Base class for objects with derived type.

    The hierarchy is structured as follows

                         AbstractObject
                                |
                 ---------------------------------
                 |                               |
              Object                       LocalObject
                 |
          CompositeObject

    Warnings
    --------
    AbstractObjects are created and managed directly by Devito.
    """

    is_AbstractObject = True

    __rargs__ = ('name', 'dtype')

    def __new__(cls, *args, **kwargs):
        with sympy_mutex:
            obj = sympy.Basic.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    def __init__(self, name, dtype):
        self.name = name
        self._dtype = dtype

    def __repr__(self):
        return self.name

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    def _hashable_content(self):
        return (self.name, self.dtype)

    @property
    def dtype(self):
        return self._dtype

    @property
    def free_symbols(self):
        return {self}

    @property
    def _C_name(self):
        return self.name

    @property
    def _C_ctype(self):
        return self.dtype

    @property
    def base(self):
        return self

    @property
    def function(self):
        return self

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class AbstractObjectWithShape(Basic, sympy.Basic, Pickable):

    """
    Base class for objects with derived type.

    The hierarchy is structured as follows

                         AbstractObjectWithShape
                                   |
                              PetscObject
    """

    AbstractObjectWithShape = True

    def __new__(cls, *args, **kwargs):

        name = kwargs.get('name')
        dtype = kwargs.get('dtype')
        dimensions, indices = cls.__indices_setup__(**kwargs)

        with sympy_mutex:
            obj = sympy.Basic.__new__(cls, *indices)

        obj._name = name
        obj._dtype = dtype
        obj._dimensions = dimensions
        obj._shape = cls.__shape_setup__(**kwargs)
        obj.__init_finalize__(*args, **kwargs)

        return obj

    def __init__(self, *args, **kwargs):
        # nothing else needs to be initalised after __new__?
        pass

    def __init_finalize__(self, *args, **kwargs):
        self._is_const = kwargs.get('is_const', False)

        # There may or may not be a `Grid`
        self._grid = kwargs.get('grid')

    @classmethod
    def __indices_setup__(cls, **kwargs):
        """Extract the object indices from ``kwargs``."""
        return (), ()

    @classmethod
    def __shape_setup__(cls, **kwargs):
        """Extract the object shape from ``kwargs``."""
        return ()

    def __repr__(self):
        return self.name

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    _ccode = _sympystr

    def _hashable_content(self):
        return (self.name, self.dtype)

    @property
    def dtype(self):
        return self._dtype

    @property
    def free_symbols(self):
        return {self}

    @property
    def _C_name(self):
        return self.name

    @property
    def _C_ctype(self):
        return self.dtype

    @property
    def is_const(self):
        return self._is_const

    @property
    def indices(self):
        """The indices of the object."""
        return DimensionTuple(*self.args, getters=self.dimensions)

    @property
    def dimensions(self):
        """Tuple of Dimensions representing the object indices."""
        return self._dimensions

    @property
    def shape(self):
        """The shape of the object."""
        return self._shape

    @property
    def ndim(self):
        """The rank of the object."""
        return len(self.indices)

    @property
    def base(self):
        return self

    @property
    def function(self):
        return self

    @property
    def grid(self):
        return self._grid

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class Object(AbstractObject, ArgProvider, Uncached):

    """
    Object with derived type defined in Python.
    """

    is_Object = True

    def __init__(self, name, dtype, value=None):
        super(Object, self).__init__(name, dtype)
        self.value = value

    __hash__ = Uncached.__hash__

    @property
    def _mem_external(self):
        return True

    @property
    def _arg_names(self):
        return (self.name,)

    def _arg_defaults(self):
        if callable(self.value):
            return {self.name: self.value()}
        else:
            return {self.name: self.value}

    def _arg_values(self, **kwargs):
        """
        Produce runtime values for this Object after evaluating user input.

        Parameters
        ----------
        **kwargs
            Dictionary of user-provided argument overrides.
        """
        if self.name in kwargs:
            obj = kwargs.pop(self.name)
            return {self.name: obj._arg_defaults()[obj.name]}
        else:
            return self._arg_defaults()


class CompositeObject(Object):

    """
    Object with composite type (e.g., a C struct) defined in Python.
    """

    __rargs__ = ('name', 'pname', 'pfields')

    def __init__(self, name, pname, pfields, value=None):
        dtype = CtypesFactory.generate(pname, pfields)
        value = self.__value_setup__(dtype, value)
        super(CompositeObject, self).__init__(name, dtype, value)

    def __value_setup__(self, dtype, value):
        return value or byref(dtype._type_())

    @property
    def pfields(self):
        return tuple(self.dtype._type_._fields_)

    @property
    def pname(self):
        return self.dtype._type_.__name__

    @property
    def fields(self):
        return [i for i, _ in self.pfields]


class LocalObject(AbstractObject):

    """
    Object with derived type defined inside an Operator.
    """

    is_LocalObject = True

    dtype = None
    """
    LocalObjects encode their dtype as a class attribute.
    """

    __rargs__ = ('name',)
    __rkwargs__ = ('cargs', 'liveness')

    def __init__(self, name, cargs=None, **kwargs):
        self.name = name
        self.cargs = as_tuple(cargs)

        self._liveness = kwargs.get('liveness', 'lazy')
        assert self._liveness in ['eager', 'lazy']

    def _hashable_content(self):
        return super()._hashable_content() + self.cargs + (self.liveness,)

    @property
    def liveness(self):
        return self._liveness

    @property
    def free_symbols(self):
        return super().free_symbols | set(self.cargs)

    @property
    def _C_init(self):
        """
        A symbolic initializer for the LocalObject, injected in the generated code.

        Notes
        -----
        To be overridden by subclasses, ignored otherwise.
        """
        return None

    @property
    def _C_free(self):
        """
        A symbolic destructor for the LocalObject, injected in the generated code.

        Notes
        -----
        To be overridden by subclasses, ignored otherwise.
        """
        return None

    @property
    def _mem_internal_eager(self):
        return self._liveness == 'eager'

    @property
    def _mem_internal_lazy(self):
        return self._liveness == 'lazy'


class PetscObject(AbstractObjectWithShape):

    is_PetscObject = True

    def __init__(self, name, dtype, **kwargs):
        self.name = name
        self._dtype = dtype
        self._is_const = kwargs.get('is_const', False)

    # taken exactly from Function class but removed the "shape_global"
    #  # and also added the option for all dimensions, shape and grid to be None
    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        dimensions = kwargs.get('dimensions')
        shape = kwargs.get('shape')

        if dimensions is None and shape is None and grid is None:
            return None

        if grid is None:
            if shape is None:
                raise TypeError("Need either `grid` or `shape`")
        elif shape is None:
            if dimensions is not None and dimensions != grid.dimensions:
                raise TypeError("Need `shape` as not all `dimensions` are in `grid`")
            shape = grid.shape_local
        elif dimensions is None:
            raise TypeError("`dimensions` required if both `grid` and "
                            "`shape` are provided")
        else:
            # Got `grid`, `dimensions`, and `shape`. We sanity-check that the
            # Dimensions in `dimensions` also appearing in `grid` have same size
            # (given by `shape`) as that provided in `grid`
            if len(shape) != len(dimensions):
                raise ValueError("`shape` and `dimensions` must have the "
                                 "same number of entries")
            loc_shape = []
            for d, s in zip(dimensions, shape):
                if d in grid.dimensions:
                    size = grid.dimension_map[d]
                    if size.glb != s and s is not None:
                        raise ValueError("Dimension `%s` is given size `%d`, "
                                         "while `grid` says `%s` has size `%d` "
                                         % (d, s, d, size.glb))
                    else:
                        loc_shape.append(size.loc)
                else:
                    loc_shape.append(s)
            shape = tuple(loc_shape)
        return shape

    # taken from Function class but removed staggered indices
    # and also added the option for all dimensions, shape and grid to be None
    @classmethod
    def __indices_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        # shape = kwargs.get('shape', None)
        shape = kwargs.get('shape')
        dimensions = kwargs.get('dimensions')

        if dimensions is None and shape is None and grid is None:
            return (), ()

        if grid is None:
            if dimensions is None:
                raise TypeError("Need either `grid` or `dimensions`")
        elif dimensions is None:
            dimensions = grid.dimensions

        return dimensions, dimensions

    @property
    def _C_ctype(self):
        """
        """
        ctypename = 'Petsc%s' % dtype_to_cstr(self.dtype).capitalize()
        ctype = dtype_to_ctype(self.dtype)
        r = type(ctypename, (ctype,), {})
        for n in range(self.ndim):
            r = POINTER(r)
        return r
