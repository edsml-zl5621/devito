from ctypes import byref, Structure, POINTER
from functools import cached_property

import sympy

from devito.tools import Pickable, as_tuple, sympy_mutex
from devito.types.args import ArgProvider
from devito.types.caching import Uncached
from devito.types.basic import Basic
from devito.types.utils import CtypesFactory
from devito.types.array import ArrayObject, ArrayBasic
from devito.types.dimension import CustomDimension

__all__ = ['Object', 'LocalObject', 'CompositeObject', 'CCompositeObject']


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
    def function(self):
        return self

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class Object(AbstractObject, ArgProvider, Uncached):

    """
    Object with derived type defined in Python.
    """

    is_Object = True

    def __init__(self, name, dtype, value=None):
        super().__init__(name, dtype)
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
        super().__init__(name, dtype, value)

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

    default_initvalue = None
    """
    The initial value may or may not be a class-level attribute. In the latter
    case, it is passed to the constructor.
    """

    __rargs__ = ('name',)
    __rkwargs__ = ('cargs', 'initvalue', 'liveness', 'is_global')

    def __init__(self, name, cargs=None, initvalue=None, liveness='lazy',
                 is_global=False, **kwargs):
        self.name = name
        self.cargs = as_tuple(cargs)
        self.initvalue = initvalue or self.default_initvalue

        assert liveness in ['eager', 'lazy']
        self._liveness = liveness

        self._is_global = is_global

    def _hashable_content(self):
        return (super()._hashable_content() +
                self.cargs +
                (self.initvalue, self.liveness, self.is_global))

    @property
    def liveness(self):
        return self._liveness

    @property
    def is_global(self):
        return self._is_global

    @property
    def free_symbols(self):
        ret = set()
        ret.update(super().free_symbols)
        for i in self.cargs:
            try:
                ret.update(i.free_symbols)
            except AttributeError:
                # E.g., pure integers
                pass
        return ret

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

    _C_modifier = None
    """
    A modifier added to the LocalObject's C declaration when the object appears
    in a function signature. For example, a subclass might define `_C_modifier = '&'`
    to impose pass-by-reference semantics.
    """

    @property
    def _mem_internal_eager(self):
        return self._liveness == 'eager'

    @property
    def _mem_internal_lazy(self):
        return self._liveness == 'lazy'

    @property
    def _mem_global(self):
        return self._is_global
    

# working
# class CCompositeObject(LocalObject):

#     """
#     Represents a composite type (e.g., a C struct) defined in C.
#     """

#     __rargs__ = ('name', 'fields')
#     __rkwargs__ = ('liveness',)

#     def __init__(self, name, fields, liveness='lazy'):
#         pfields = [(i._C_name, i._C_ctype) for i in fields]
#         self.__class__.dtype = type('MatContext', (Structure,), {'_fields_': pfields})
#         super().__init__(name, cargs=None, initvalue=None, liveness=liveness)
#         self._pfields = pfields
#         self._fields = fields

#     @property
#     def pfields(self):
#         return self._pfields

#     @property
#     def fields(self):
#         return self._fields
    
#     @property
#     def _C_ctype(self):
#         return POINTER(self.dtype) if self.liveness == \
#             'eager' else self.dtype

#     _C_modifier = ' *'
    


class CCompositeObject(ArrayObject):

    # Not a performance-sensitive object
    _data_alignment = False

    @classmethod
    def __indices_setup__(cls, **kwargs):
        try:
            return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
        except KeyError:
            nthreads = kwargs['npthreads']
            dim = CustomDimension(name='wi', symbolic_size=nthreads)
            return (dim,), (dim,)

    @property
    def dim(self):
        assert len(self.dimensions) == 1
        return self.dimensions[0]

    @property
    def npthreads(self):
        return self.dim.symbolic_size

    @property
    def index(self):
        if self.size == 1:
            return 0
        else:
            return self.dim



# class CCompositeObject(ArrayObject):

#     @classmethod
#     def __dtype_setup__(cls, **kwargs):
#         pname = kwargs.get('pname', 't%s' % kwargs['name'])
#         pfields = cls.__pfields_setup__(**kwargs)
#         return type(pname, (Structure,), {'_fields_': pfields})
    
#     @classmethod
#     def __indices_setup__(cls, **kwargs):
#         try:
#             return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
#         except KeyError:
#             # num = kwargs['num']
#             dim = CustomDimension(name='wi', symbolic_size=5)
#             return (dim,), (dim,)
    
#     _C_modifier = ' *'


    # __rargs__ = ('name', 'usr_ctx')
    # __rkwargs__ = ('liveness',)

    # def __init__(self, name, usr_ctx, liveness='lazy'):
    #     # pfields = [(i._C_name, dtype_to_ctype(i.dtype))
    #     #                for i in usr_ctx if isinstance(i, Basic)]
    #     # ctype = dtype_to_ctype(i.dtype)
    #     pfields = [(i._C_name, i._C_ctype)
    #                    for i in usr_ctx]
    #     # from IPython import embed; embed()
    #     self.__class__.dtype = type('MatContext', (Structure,), {'_fields_': pfields})
    #     super().__init__(name, cargs=None, initvalue=None, liveness=liveness)
    #     self._pfields = pfields
    #     self._usr_ctx = usr_ctx

    #     # assert liveness in ['eager', 'lazy']
    #     # self._liveness = liveness

    # @property
    # def pfields(self):
    #     return self._pfields

    # @property
    # def usr_ctx(self):
    #     return self._usr_ctx
    
    # @property
    # def _C_ctype(self):
    #     return POINTER(self.dtype) if self.liveness == \
    #         'eager' else self.dtype

    # _C_modifier = ' *'
