from devito.ir.iet import Expression
from devito.ir.equations import OpMatVec, OpRHS


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
