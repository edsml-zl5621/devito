import sympy

from devito.tools import Reconstructable, sympy_mutex


class BaseInfoExpr(sympy.Function, Reconstructable):
    __rargs__ = ('expr',)

    def __new__(cls, expr, **kwargs):
        with sympy_mutex:
            obj = sympy.Basic.__new__(cls, expr)
        obj._expr = expr

        for key, value in kwargs.items():
            setattr(obj, "_%s" % key, value)

        return obj

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.expr)

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    def __hash__(self):
        return hash(self.expr)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.expr == other.expr

    @property
    def expr(self):
        return self._expr

    func = Reconstructable._rebuild


class LinearSolveExpr(BaseInfoExpr):

    __rkwargs__ = ('target', 'solver_parameters', 'matvecs',
                   'formfuncs', 'formrhs', 'arrays')

    defaults = {
        'ksp_type': 'gmres',
        'pc_type': 'jacobi',
        'ksp_rtol': 1e-7,
        'ksp_atol': 1e-50,
        'ksp_divtol': 1e4,
        'ksp_max_it': 10000
    }

    def __new__(cls, expr, target=None, solver_parameters=None,
                matvecs=None, formfuncs=None, formrhs=None, arrays=None, **kwargs):

        if solver_parameters is None:
            solver_parameters = cls.defaults
        else:
            for key, val in cls.defaults.items():
                solver_parameters[key] = solver_parameters.get(key, val)

        return super().__new__(cls, expr, target=target, solver_parameters=solver_parameters,
                               matvecs=matvecs, formfuncs=formfuncs, formrhs=formrhs,
                               arrays=arrays, **kwargs)

    @property
    def target(self):
        return self._target

    @property
    def solver_parameters(self):
        return self._solver_parameters

    @property
    def matvecs(self):
        return self._matvecs

    @property
    def formfuncs(self):
        return self._formfuncs

    @property
    def formrhs(self):
        return self._formrhs

    @property
    def arrays(self):
        return self._arrays


class CallbackExpr(BaseInfoExpr):
    __rkwargs__ = ('parent_modulo_dims',)

    def __new__(cls, expr, parent_modulo_dims=None, **kwargs):
        return super().__new__(cls, expr,
                               parent_modulo_dims=parent_modulo_dims,
                               **kwargs)

    @property
    def parent_modulo_dims(self):
        return self._parent_modulo_dims
    

class CallbackExprExpr(sympy.Function):
    @classmethod
    def eval(cls, true_rhs, *t_args):
        return None

