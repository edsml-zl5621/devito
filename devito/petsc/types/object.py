from ctypes import POINTER
import ctypes

from devito.tools import CustomDtype, dtype_to_cstr, as_tuple
from devito.types import LocalObject, CCompositeObject, ModuloDimension, TimeDimension, ArrayObject, CustomDimension, PointerArray
from devito.symbolics import Byref

from devito.petsc.iet.utils import petsc_call


class DM(LocalObject):
    """
    PETSc Data Management object (DM). This is the primary DM instance
    created within the main kernel and linked to the SNES
    solver using `SNESSetDM`.
    """
    dtype = CustomDtype('DM')

    def __init__(self, *args, stencil_width=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._stencil_width = stencil_width

    @property
    def stencil_width(self):
        return self._stencil_width

    @property
    def _C_free(self):
        return petsc_call('DMDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 3


class CallbackDM(LocalObject):
    """
    PETSc Data Management object (DM). This is the DM instance
    accessed within the callback functions via `SNESGetDM`.
    """
    dtype = CustomDtype('DM')

    def __init__(self, *args, stencil_width=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._stencil_width = stencil_width

    @property
    def stencil_width(self):
        return self._stencil_width


class Mat(LocalObject):
    """
    PETSc Matrix object (Mat).
    """
    dtype = CustomDtype('Mat')

    @property
    def _C_free(self):
        return petsc_call('MatDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 1


class LocalMat(LocalObject):
    dtype = CustomDtype('Mat')


class LocalVec(LocalObject):
    """
    PETSc local vector object (Vec).
    A local vector has ghost locations that contain values that are
    owned by other MPI ranks.
    """
    dtype = CustomDtype('Vec')


class GlobalVec(LocalObject):
    """
    PETSc global vector object (Vec).
    A global vector is a parallel vector that has no duplicate values
    between MPI ranks. A global vector has no ghost locations.
    """
    dtype = CustomDtype('Vec')

    @property
    def _C_free(self):
        return petsc_call('VecDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 0


class PetscMPIInt(LocalObject):
    """
    PETSc datatype used to represent `int` parameters
    to MPI functions.
    """
    dtype = CustomDtype('PetscMPIInt')


class PetscInt(LocalObject):
    """
    PETSc datatype used to represent `int` parameters
    to PETSc functions.
    """
    dtype = CustomDtype('PetscInt')

    # @property
    # def _C_ctype(self):
    #     return ctypes.c_int


class KSP(LocalObject):
    """
    PETSc KSP : Linear Systems Solvers.
    Manages Krylov Methods.
    """
    dtype = CustomDtype('KSP')


class SNES(LocalObject):
    """
    PETSc SNES : Non-Linear Systems Solvers.
    """
    dtype = CustomDtype('SNES')

    @property
    def _C_free(self):
        return petsc_call('SNESDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 2


class PC(LocalObject):
    """
    PETSc object that manages all preconditioners (PC).
    """
    dtype = CustomDtype('PC')


class KSPConvergedReason(LocalObject):
    """
    PETSc object - reason a Krylov method was determined
    to have converged or diverged.
    """
    dtype = CustomDtype('KSPConvergedReason')


class DMDALocalInfo(LocalObject):
    """
    PETSc object - C struct containing information
    about the local grid.
    """
    dtype = CustomDtype('DMDALocalInfo')


class PetscErrorCode(LocalObject):
    """
    PETSc datatype used to return PETSc error codes.
    https://petsc.org/release/manualpages/Sys/PetscErrorCode/
    """
    dtype = CustomDtype('PetscErrorCode')


class DummyArg(LocalObject):
    """
    A void pointer used to satisfy the function
    signature of the `FormFunction` callback.
    """
    dtype = CustomDtype('void', modifier='*')


class PETScStruct(CCompositeObject):

    __rargs__ = ('name', 'pname', 'fields')

    def __init__(self, name, pname, fields, modifier=None, liveness='lazy'):
        pfields = [(i._C_name, i._C_ctype) for i in fields]
        super().__init__(name, pname, pfields, modifier, liveness)
        self._fields = fields

    @property
    def fields(self):
        return self._fields

    @property
    def time_dim_fields(self):
        """
        Fields within the struct that are updated during the time loop.
        These are not set in the `PopulateMatContext` callback.
        """
        return [f for f in self.fields
                if isinstance(f, (ModuloDimension, TimeDimension))]

    @property
    def callback_fields(self):
        """
        Fields within the struct that are initialized in the `PopulateMatContext`
        callback. These fields are not updated in the time loop.
        """
        return [f for f in self.fields if f not in self.time_dim_fields]

    # TODO: reevaluate this, does it even ever go here?
    # @property
    # def _C_ctype(self):
    #     # from IPython import embed; embed()
    #     return POINTER(self.dtype) if self.liveness == \
    #         'eager' else self.dtype

    _C_modifier = ' *'

    # TODO: maybe this should move to CCompositeObject itself
    @property
    def _fields_(self):
        return [(i._C_name, i._C_ctype) for i in self.fields]

    # IMPROVE: this is because of the use inside iet/visitors struct decl 
    @property
    def __name__(self):
        return self.pname



class StartPtr(LocalObject):
    def __init__(self, name, dtype):
        super().__init__(name=name)
        self.dtype = CustomDtype(dtype_to_cstr(dtype), modifier=' *')




class SingleIS(LocalObject):
    """
    """
    dtype = CustomDtype('IS')



################################### rethink ALL BELOW since they are just ptrs to already exisiting classes e.g Mat *submats....
# need to be able to index them though etc...
# TODO may have to re-think this, not sure if quite right -> CREATE A BASE CLASS FOR 
#  ALL OBJECTS WHICH APPEAR AS A *PTR and then need to be indexed into to destroy them i.e each element of the array
class IS(ArrayObject):
    """
    Index set object used for efficient indexing into vectors and matrices.
    https://petsc.org/release/manualpages/IS/IS/
    """
    _data_alignment = False

    def __init_finalize__(self, *args, **kwargs):
        self._nindices = kwargs.pop('nindices', ())
        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        try:
            return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
        except KeyError:
            nindices = kwargs.get('nindices', ())
            dim = CustomDimension(name='d', symbolic_size=nindices)
            return (dim,), (dim,)

    @property
    def dim(self):
        assert len(self.dimensions) == 1
        return self.dimensions[0]

    @property
    def nindices(self):
        return self._nindices

    @property
    def dtype(self):
        return CustomDtype('IS', modifier=' *')

    @property
    def _C_name(self):
        return self.name

    @property
    def _mem_stack(self):
        return False

    # @property
    # def _C_ctype(self):
    #     return ctypes.c_void_p

    @property
    def _C_free(self):
        destroy_calls = [
            petsc_call('ISDestroy', [Byref(self.indexify().subs({self.dim: i}))])
            for i in range(self._nindices)
        ]
        destroy_calls.append(petsc_call('PetscFree', [self.function]))
        return destroy_calls


class SubDM(ArrayObject):

    _data_alignment = False

    def __init_finalize__(self, *args, **kwargs):
        self._nindices = kwargs.pop('nindices', ())
        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        try:
            return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
        except KeyError:
            nindices = kwargs.get('nindices', ())
            dim = CustomDimension(name='d', symbolic_size=nindices)
            return (dim,), (dim,)

    @property
    def dim(self):
        assert len(self.dimensions) == 1
        return self.dimensions[0]

    @property
    def nindices(self):
        return self._nindices

    @property
    def dtype(self):
        return CustomDtype('DM', modifier=' *')

    @property
    def _C_name(self):
        return self.name

    @property
    def _mem_stack(self):
        return False

    # NOTE ADD THE FUNCTIONALITY SO THAT ARRAYOBJECTS CAN BE DESTROYED .. or re-think this class
    @property
    def _C_free(self):
        destroy_calls = [
            petsc_call('DMDestroy', [Byref(self.indexify().subs({self.dim: i}))])
            for i in range(self._nindices)
        ]
        destroy_calls.append(petsc_call('PetscFree', [self.function]))
        return destroy_calls



# class SubMats(ArrayObject):

#     _data_alignment = False

#     def __init_finalize__(self, *args, **kwargs):
#         self._nindices = kwargs.pop('nindices', ())
#         super().__init_finalize__(*args, **kwargs)

#     @classmethod
#     def __indices_setup__(cls, **kwargs):
#         try:
#             return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
#         except KeyError:
#             nindices = kwargs.get('nindices', ())
#             dim = CustomDimension(name='d', symbolic_size=nindices)
#             return (dim,), (dim,)

#     @property
#     def dim(self):
#         assert len(self.dimensions) == 1
#         return self.dimensions[0]

#     _C_modifier = ' *'

#     @property
#     def nindices(self):
#         return self._nindices

#     @property
#     def dtype(self):
#         return CustomDtype('Mat', modifier=' *')

#     @property
#     def _C_name(self):
#         return self.name

#     @property
#     def _mem_stack(self):
#         return False


class SubMats(ArrayObject):
    _data_alignment = False

    def __init_finalize__(self, *args, **kwargs):
        self._nindices = kwargs.pop('nindices', ())
        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        try:
            return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
        except KeyError:
            nindices = kwargs.get('nindices', ())
            dim = CustomDimension(name='d', symbolic_size=nindices)
            return (dim,), (dim,)

    @property
    def dim(self):
        assert len(self.dimensions) == 1
        return self.dimensions[0]

    _C_modifier = ' *'

    @property
    def nindices(self):
        return self._nindices

    @property
    def dtype(self):
        return CustomDtype('Mat', modifier=' *')

    @property
    def _C_name(self):
        return self.name

    @property
    def _mem_stack(self):
        return False
