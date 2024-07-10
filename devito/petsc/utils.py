import os

from devito.ir.equations import OpInjectSolve
from devito.tools import memoized_func
from devito.ir.iet import Call, FindSymbols
from devito.petsc.iet.nodes import PETScCallable, InjectSolveDummy
from devito.petsc.types import PETScStruct

# Mapping special Eq operations to their corresponding IET Expression subclass types.
# These operations correspond to subclasses of Eq utilised within PETScSolve.
petsc_iet_mapper = {OpInjectSolve: InjectSolveDummy}


solver_mapper = {
    'gmres': 'KSPGMRES',
    'jacobi': 'PCJACOBI',
    None: 'PCNONE'
}


@memoized_func
def get_petsc_dir():
    # *** First try: via commonly used environment variables
    for i in ['PETSC_DIR']:
        petsc_dir = os.environ.get(i)
        if petsc_dir:
            return petsc_dir
    # TODO: Raise error if PETSC_DIR is not set
    return None


@memoized_func
def get_petsc_arch():
    # *** First try: via commonly used environment variables
    for i in ['PETSC_ARCH']:
        petsc_arch = os.environ.get(i)
        if petsc_arch:
            return petsc_arch
    # TODO: Raise error if PETSC_ARCH is not set
    return None


def core_metadata():
    petsc_dir = get_petsc_dir()
    petsc_arch = get_petsc_arch()

    # Include directories
    global_include = os.path.join(petsc_dir, 'include')
    config_specific_include = os.path.join(petsc_dir, f'{petsc_arch}', 'include')
    include_dirs = (global_include, config_specific_include)

    # Lib directories
    lib_dir = os.path.join(petsc_dir, f'{petsc_arch}', 'lib')

    return {
        'includes': ('petscksp.h', 'petscsnes.h', 'petscdmda.h'),
        'include_dirs': include_dirs,
        'libs': ('petsc'),
        'lib_dirs': lib_dir,
        'ldflags': ('-Wl,-rpath,%s' % lib_dir)
    }
