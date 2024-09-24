from functools import singledispatch

import sympy
import numpy as np

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.derivative import Derivative
from devito.types import Eq, Constant
from devito.types.equation import InjectSolveEq
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions, uxreplace
from devito.petsc.types import LinearSolveExpr, PETScArray, CallbackExpr
from devito.symbolics import retrieve_functions, INT
from devito.ir import SymbolRegistry


__all__ = ['PETScSolve', 'EssentialBC']


class EssentialBC(Eq):
    pass


def PETScSolve(eqns, target, solver_parameters=None, **kwargs):

    sregistry = SymbolRegistry()
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
    # TODO: adjust this for all eqns
    funcs_placeholder = set(retrieve_functions(eqns[0].lhs - eqns[0].rhs))
    funcs_placeholder = list(funcs_placeholder)
    # from IPython import embed; embed()
    # TODO: fix modulo dimension thing i.e rebuild the callbackexpr later on or use mapper technique
    funcs_placeholder.append(funcs_placeholder[-1].backward)
    funcs = list(set(retrieve_functions(eqns)))
    # from IPython import embed; embed()

    for eq in eqns:
        b, F_target, target_funcs = separate_eqn(eq, target, sregistry)
        # TODO: Current assumption is that problem is linear and user has not provided
        # a jacobian. Hence, we can use F_target to form the jac-vec product

        if isinstance(eq, EssentialBC):
            # TODO: FIX THIS-> RE-EVALUATE
            btmp, Ftmp, _ = separate_eqn(eqns[0], target, sregistry)
            centre = centre_stencil(Ftmp, target)
            matvecs.append(Eq(
                arrays['y_matvec'],
                CallbackExpr(centre.subs(generate_mapper(arrays['x_matvec'], target_funcs))),
                subdomain=eq.subdomain
            ))

            formfuncs.append(Eq(
            arrays['y_formfunc'],
            CallbackExpr(Constant(name=sregistry.make_name(prefix='zero'), value=0., dtype=target.grid.dtype)),
            subdomain=eq.subdomain
            ))
            
            formrhs.append(Eq(
                arrays['b_tmp'],
                CallbackExpr(Constant(name=sregistry.make_name(prefix='zero'), value=0., dtype=target.grid.dtype)),
                subdomain=eq.subdomain
            ))

        else:

            matvecs.append(Eq(
                arrays['y_matvec'],
                CallbackExpr(F_target.subs(generate_mapper(arrays['x_matvec'], target_funcs))),
                subdomain=eq.subdomain
            ))

            formfuncs.append(Eq(
                arrays['y_formfunc'],
                CallbackExpr(F_target.subs(generate_mapper(arrays['x_formfunc'], target_funcs))),
                subdomain=eq.subdomain
            ))
            # from IPython import embed; embed()
            formrhs.append(Eq(
                arrays['b_tmp'],
                CallbackExpr(b),
                subdomain=eq.subdomain
            ))
    # from IPython import embed; embed()
    # Placeholder equation for inserting calls to the solver
    # TODO: dummy_expr should include functions from all equations that appear
    # in the set of equations passed into PETScSolve
    dummy_expr = sum(funcs_placeholder)
    # TODO: pass CallbackExpr through initial lowering to generate correct modulo dims 
    inject_solve = InjectSolveEq(target, LinearSolveExpr(
        dummy_expr, target=target, solver_parameters=solver_parameters,
        matvecs=matvecs, formfuncs=formfuncs,
        formrhs=formrhs, arrays=arrays,
    ), subdomain=eq.subdomain)


    return [inject_solve]


def separate_eqn(eqn, target, sregistry):
    """
    Separate the eqn into two separate expressions,
    where F(target) = b.
    """
    zeroed_eqn = Eq(eqn.lhs - eqn.rhs, Constant(name=sregistry.make_name(prefix='zero'), value=0., dtype=eqn.lhs.dtype))
    zeroed_eqn = eval_time_derivatives(zeroed_eqn.lhs)
    target_funcs  = set(spatial_targets(zeroed_eqn, target))
    b, F_target = remove_target(zeroed_eqn, target_funcs)
    return -b, F_target, target_funcs


def spatial_targets(eq, target):
    funcs = retrieve_functions(eq)

    if any(dim.is_Time for dim in target.dimensions):
        time_idx_target = [i for i, d in zip(target.indices, target.dimensions) if d.is_Time]
        assert len(time_idx_target) == 1
        funcs_to_subs = [
            func for func in funcs 
            if func.function is target.function and time_idx_target[0] 
            in func.indices
        ]
    else:
        funcs_to_subs = [
            func for func in funcs 
            if func.function is target.function
        ]

    return funcs_to_subs


def generate_mapper(array, funcs_to_subs):
    space_indices = [
        tuple(i for i, d in zip(func.indices, func.dimensions) if d.is_Space) for func in funcs_to_subs
    ]
    array_lst = [array.subs({ai: fi for ai, fi in zip(array.indices, indices)}) for indices in space_indices]
    return {func: arr for func, arr in zip(funcs_to_subs, array_lst)}


@singledispatch
def remove_target(expr, targets):
    return (0, expr) if expr in targets else (expr, 0)


@remove_target.register(sympy.Add)
def _(expr, targets):
    if not any(expr.has(t) for t in targets):
        return (expr, 0)

    args_b, args_F = zip(*(remove_target(a, targets) for a in expr.args))
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_target.register(Mul)
def _(expr, targets):
    if not any(expr.has(t) for t in targets):
        return (expr, 0)

    args_b, args_F = zip(*[remove_target(a, targets) if any(a.has(t) for t in targets)
                           else (a, a) for a in expr.args])
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_target.register(Derivative)
def _(expr, target):
    return (0, expr) if any(expr.has(t) for t in target) else (expr, 0)
    # return (0, expr) if expr.has(target) else (expr, 0)


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
