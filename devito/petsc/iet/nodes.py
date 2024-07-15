from operator import attrgetter
from functools import cached_property

from devito.ir.iet import Expression, Callable, FindSymbols
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

    # @cached_property
    # def dimensions(self):
    #     # ret = set().union(*[d._defines for d in self._dimensions])

    #     # During compilation other Dimensions may have been produced
    #     dimensions = FindSymbols('dimensions').visit(self)

    #     # from IPython import embed; embed()

    #     # ret.update(d for d in dimensions if d.is_PerfKnob)

    #     # ret = tuple(sorted(ret, key=attrgetter('name')))

    #     return dimensions

    @property
    def unused_parameters(self):
        return self._unused_parameters
    


