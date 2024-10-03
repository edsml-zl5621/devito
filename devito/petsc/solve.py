from functools import singledispatch

import sympy

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.derivative import Derivative
from devito.types import Eq, Symbol, SteppingDimension
from devito.types.equation import InjectSolveEq
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions
from devito.tools import as_tuple
from devito.petsc.types import LinearSolveExpr, PETScArray


__all__ = ['PETScSolve']


def PETScSolve(eqns, target, solver_parameters=None, **kwargs):
    prefixes = ['y_matvec', 'x_matvec', 'y_formfunc', 'x_formfunc', 'b_tmp']

    arrays = {
        p: PETScArray(name='%s_%s' % (p, target.name),
                           dtype=target.dtype,
                           dimensions=target.space_dimensions,
                           shape=target.grid.shape,
                           liveness='eager',
                           halo=[target.halo[d] for d in target.space_dimensions],
                           space_order=target.space_order)
        for p in prefixes
    }

    matvecs = []
    formfuncs = []
    formrhs = []

    eqns = as_tuple(eqns)
    funcs = retrieve_functions(eqns)
    time_mapper = generate_time_mapper(funcs)

    for eq in eqns:
        b, F_target = separate_eqn(eq, target)
        b, F_target = b.subs(time_mapper), F_target.subs(time_mapper)

        # TODO: Current assumption is that problem is linear and user has not provided
        # a jacobian. Hence, we can use F_target to form the jac-vec product
        matvecs.append(Eq(
            arrays['y_matvec'],
            F_target.subs({target: arrays['x_matvec']}),
            subdomain=eq.subdomain
        ))

        formfuncs.append(Eq(
            arrays['y_formfunc'],
            F_target.subs({target: arrays['x_formfunc']}),
            subdomain=eq.subdomain
        ))

        formrhs.append(Eq(
            arrays['b_tmp'],
            b,
            subdomain=eq.subdomain
        ))

    # Placeholder equation for inserting calls to the solver and generating
    # correct time loop etc
    inject_solve = InjectSolveEq(target, LinearSolveExpr(
        expr=tuple(funcs),
        target=target,
        solver_parameters=solver_parameters,
        matvecs=matvecs,
        formfuncs=formfuncs,
        formrhs=formrhs,
        arrays=arrays,
        time_mapper=time_mapper,
    ), subdomain=eq.subdomain)

    return [inject_solve]


def separate_eqn(eqn, target):
    """
    Separate the equation into two separate expressions,
    where F(target) = b.
    """
    zeroed_eqn = Eq(eqn.lhs - eqn.rhs, 0)
    tmp = eval_time_derivatives(zeroed_eqn.lhs)
    b, F_target = remove_target(tmp, target)
    return -b, F_target


@singledispatch
def remove_target(expr, target):
    return (0, expr) if expr == target else (expr, 0)


@remove_target.register(sympy.Add)
def _(expr, target):
    if not expr.has(target):
        return (expr, 0)

    args_b, args_F = zip(*(remove_target(a, target) for a in expr.args))
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_target.register(Mul)
def _(expr, target):
    if not expr.has(target):
        return (expr, 0)

    args_b, args_F = zip(*[remove_target(a, target) if a.has(target)
                           else (a, a) for a in expr.args])
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_target.register(Derivative)
def _(expr, target):
    return (0, expr) if expr.has(target) else (expr, 0)


@singledispatch
def centre_stencil(expr, target):
    """
    Extract the centre stencil from an expression. Its coefficient is what
    would appear on the diagonal of the matrix system if the matrix were
    formed explicitly.
    """
    return expr if expr == target else 0


@centre_stencil.register(sympy.Add)
def _(expr, target):
    if not expr.has(target):
        return 0

    args = [centre_stencil(a, target) for a in expr.args]
    return expr.func(*args, evaluate=False)


@centre_stencil.register(Mul)
def _(expr, target):
    if not expr.has(target):
        return 0

    args = []
    for a in expr.args:
        if not a.has(target):
            args.append(a)
        else:
            args.append(centre_stencil(a, target))

    return expr.func(*args, evaluate=False)


@centre_stencil.register(Derivative)
def _(expr, target):
    if not expr.has(target):
        return 0
    args = [centre_stencil(a, target) for a in expr.evaluate.args]
    return expr.evaluate.func(*args)


def generate_time_mapper(funcs):
    """
    Replace time indices with `Symbols` in equations used within
    PETSc callback functions. These symbols are Uxreplaced at the IET
    level to align with the `TimeDimension` and `ModuloDimension` objects
    present in the inital lowering.
    NOTE: All functions used in PETSc callback functions are attached to
    the `LinearSolveExpr` object, which is passed through the initial lowering
    (and subsequently dropped and replaced with calls to run the solver).
    Therefore, the appropriate time loop will always be correctly generated inside
    the main kernel.
    """
    time_indices = list({
        i if isinstance(d, SteppingDimension) else d
        for f in funcs
        for i, d in zip(f.indices, f.dimensions)
        if d.is_Time
    })
    tau_symbs = [Symbol('tau%d' % i) for i in range(len(time_indices))]
    return {time: tau for time, tau in zip(time_indices, tau_symbs)}
