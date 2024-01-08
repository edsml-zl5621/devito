from devito.passes.iet.engine import iet_pass, FindNodes
from devito.ir.iet import Expression


__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    tmp = FindNodes(Expression).visit(iet)

    action = [i for i in tmp if i.expr.is_action]


    # add necessary include directories for petsc (TEMP)
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
    kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/arch-linux-c-debug/include')
    kwargs['compiler'].add_libraries('petsc')
    libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
    kwargs['compiler'].add_library_dirs(libdir)
    kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)
    return iet, {'includes': ['petscksp.h', 'petscdmda.h']}