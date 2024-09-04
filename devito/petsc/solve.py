from functools import singledispatch

import sympy

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.derivative import Derivative
from devito.types import Eq
from devito.types.equation import InjectSolveEq
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions, uxreplace
from devito.petsc.types import LinearSolveExpr, PETScArray, CallbackExpr
from devito.symbolics import retrieve_functions, INT


__all__ = ['PETScSolve', 'NaturalBC', 'EssentialBC']


class NaturalBC(Eq):
    pass


class EssentialBC(Eq):
    pass


def PETScSolve(eq, target, bcs=None, solver_parameters=None, **kwargs):
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

    b, F_target = separate_eqn(eq, target)

    # Passed through main kernel and removed at iet level, used to generate
    # correct time loop etc
    dummy = list(set(retrieve_functions(F_target - b)))
    dummy_expr = sum(dummy)

    # TODO: Current assumption is that problem is linear and user has not provided
    # a jacobian. Hence, we can use F_target to form the jac-vec product

    matvecaction = Eq(
        arrays['y_matvec'],
        CallbackExpr(F_target.subs({target: arrays['x_matvec']}), *dummy),
        subdomain=eq.subdomain
    )

    formfunction = Eq(
        arrays['y_formfunc'],
        CallbackExpr(F_target.subs({target: arrays['x_formfunc']}), *dummy),
        subdomain=eq.subdomain
    )

    rhs = Eq(
        arrays['b_tmp'],
        CallbackExpr(b, *dummy),
        subdomain=eq.subdomain
    )

    # Placeholder equation for inserting calls to the solver
    inject_solve = InjectSolveEq(target, LinearSolveExpr(
        dummy_expr, target=target, solver_parameters=solver_parameters,
        matvecs=[matvecaction], formfuncs=[formfunction],
        formrhs=[rhs], arrays=arrays,
    ), subdomain=eq.subdomain)

    if not bcs:
        return [inject_solve]

    # NOTE: BELOW IS NOT FULLY TESTED/IMPLEMENTED YET
    bcs_for_matvec = []
    bcs_for_formfunc = []
    bcs_for_rhs = []

    # OBVIOUSLY FIX THIS:
    rhs = Eq(
        arrays['b_tmp'],
        CallbackExpr(b, *dummy),
        subdomain=target.grid.subdomains['domain']    
    )

    centre = centre_stencil(F_target, target)
    # for bc in bcs:


        # if isinstance(bc, EssentialBC):
        #     bcs_for_rhs.append(Eq(arrays['b_tmp'], CallbackExpr(0., *dummy), subdomain=bc.subdomain))
        #     bcs_for_formfunc.append(Eq(arrays['y_formfunc'], CallbackExpr(0., *dummy), subdomain=bc.subdomain))
        #     centre = uxreplace(centre, {target: arrays['x_matvec']})
        #     bcs_for_matvec.append(Eq(
        #         arrays['y_matvec'],
        #         CallbackExpr(centre, *dummy),
        #         subdomain=bc.subdomain
        #     ))

        # elif isinstance(bc, NaturalBC):
        #     # new_rhs = F_target.subs({target: arrays['x_formfunc']})
        #     tmp = neumann(Eq(target, F_target, subdomain=bc.subdomain), bc.subdomain, bc.subdomain.name, array=arrays['x_formfunc'])
        #     bcs_for_formfunc.append(Eq(arrays['y_formfunc'],
        #                                CallbackExpr(tmp), subdomain=bc.subdomain))



    inject_solve = InjectSolveEq(target, LinearSolveExpr(
        dummy_expr, target=target, solver_parameters=solver_parameters,
        matvecs=[matvecaction]+bcs_for_matvec,
        formfuncs=[formfunction],
        formrhs=[rhs],
        arrays=arrays,
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
