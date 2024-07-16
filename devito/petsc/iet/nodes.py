from operator import attrgetter
from functools import cached_property

from devito.ir.iet import Expression, Callable, FindSymbols, Call
from devito.ir.equations import OpMatVec, OpRHS
from devito.tools import as_tuple


class LinearSolverExpression(Expression):

    """
    Base class for general expressions required by a
    matrix-free linear solve of the form Ax=b.
    """
    pass


class MatVecAction(LinearSolverExpression):

    """
    Expression representing matrix-vector multiplication.
    """

    def __init__(self, expr, pragmas=None, operation=OpMatVec):
        super().__init__(expr, pragmas=pragmas, operation=operation)


class RHSLinearSystem(LinearSolverExpression):

    """
    Expression to build the RHS of a linear system.
    """

    def __init__(self, expr, pragmas=None, operation=OpRHS):
        super().__init__(expr, pragmas=pragmas, operation=operation)

    
class PETScCallable(Callable):

    def __init__(self, name, body, retval=None, parameters=None, prefix=None, unused_parameters=None):
        super().__init__(name, body, retval, parameters, prefix)
        self._unused_parameters = as_tuple(unused_parameters)

    @property
    def unused_parameters(self):
        return self._unused_parameters
    

class Callback(Call):
    """
    Callback as a function pointer.

    Parameters
    ----------
    name : str
        The name of the callback.
    retval : str
        The return type of the callback.
    param_types : str or list of str
        The return type for each argument of the callback.

    Notes
    -----
    The reason Callback is an IET type rather than a SymPy type is
    due to the fact that, when represented at the SymPy level, the IET
    engine fails to bind the callback to a specific Call. Consequently,
    errors occur during the creation of the call graph.
    """
    # TODO: Create a common base class for Call and Callback to avoid
    # having arguments=None here
    def __init__(self, name, retval, param_types, arguments=None):
        super().__init__(name=name)
        self.retval = retval
        self.param_types = as_tuple(param_types)
    
    @property
    def callback_form(self):
        return "%s" % self.name
    

class MatVecCallback(Callback):
    @property
    def callback_form(self):
        param_types_str = ', '.join([str(t) for t in self.param_types])
        return "(%s (*)(%s))%s" % (self.retval, param_types_str, self.name)
