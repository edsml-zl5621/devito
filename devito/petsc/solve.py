from functools import singledispatch

import sympy

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.derivative import Derivative
from devito.types import Eq, Symbol, SteppingDimension, Function
from devito.types.equation import InjectSolveEq
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions
from devito.tools import as_tuple
from devito.petsc.types import (LinearSolver, PETScArray, DM,
                                FieldData, FieldDataNest, DMComposite)


__all__ = ['PETScSolve']


def PETScSolve(eqns_targets, target=None, solver_parameters=None, **kwargs):
    """
    Linear PETSc solver
    """
    if target is not None:
        eqns_targets = {target: eqns_targets}

    # TODO: If target is a vector, also default to MATNEST (same as FD)
    if len(eqns_targets.keys()) == 1:
        target, eqns = next(iter(eqns_targets.items())) 
        eqns = as_tuple(eqns)
        funcs = retrieve_functions(eqns)
        time_mapper = generate_time_mapper(funcs)
        field_data = generate_field_solve(eqns, target, time_mapper)
        # Placeholder equation for inserting calls to the solver and generating
        # correct time loop etc.
        inject_solve = InjectSolveEq(
            target,
            LinearSolver(set(funcs), solver_parameters,
            fielddata=field_data, parent_dm=field_data.dmda,
            time_mapper=time_mapper)
        )
        return [inject_solve]
    # NEST
    else:
        # from IPython import embed; embed()
        # TODO : improve this, probs move whole matnest bit into separate function
        combined_eqns = [item for sublist in eqns_targets.values() for item in sublist]
        funcs = retrieve_functions(combined_eqns)
        time_mapper = generate_time_mapper(funcs)
        nest = FieldDataNest()
        children_dms = []

        for target, eqns in eqns_targets.items():
            eqns = as_tuple(eqns)
            field_data = generate_field_solve(eqns, target, time_mapper)
            nest.add_field_data(field_data)
            children_dms.append(field_data.dmda)

        targets = list(eqns_targets.keys())
        parent_dm = DMComposite(name='da_%s' % '_'.join(t.name for t in targets), targets=targets)

        # Can place any target within targets on the lhs here
        inject_solve = InjectSolveEq(
            target,
            LinearSolver(set(funcs),
            solver_parameters, fielddata=nest, parent_dm=parent_dm,
            children_dms=children_dms, time_mapper=time_mapper)
        )
        return [inject_solve]


def generate_field_solve(eqns, target, time_mapper):
    #TODO: change these names
    prefixes = ['y_matvec', 'x_matvec', 'y_formfunc', 'x_formfunc', 'b_tmp']

    # field DMDA
    dmda = DM(
        name='da_%s' % target.name, liveness='eager', target=target
    )
    arrays = {
        '%s_%s' % (p, target.name): PETScArray(name='%s_%s' % (p, target.name),
                           dtype=target.dtype,
                           dimensions=target.space_dimensions,
                           shape=target.grid.shape,
                           liveness='eager',
                           halo=[target.halo[d] for d in target.space_dimensions],
                           space_order=target.space_order,
                           dmda=dmda)
        for p in prefixes
    }

    matvecs = []
    formfuncs = []
    formrhs = []

    for eq in eqns:
        b, F_target, targets = separate_eqn(eq, target)

        # TODO: Current assumption is that problem is linear and user has not provided
        # a jacobian. Hence, we can use F_target to form the jac-vec product
        matvecs.append(Eq(
            arrays['y_matvec_%s' % target.name],
            F_target.subs({target: arrays['x_matvec_%s' % target.name]}),
            subdomain=eq.subdomain
        ))

        formfuncs.append(Eq(
            arrays['y_formfunc_%s' % target.name],
            F_target.subs({target: arrays['x_formfunc_%s' % target.name]}),
            subdomain=eq.subdomain
        ))

        formrhs.append(Eq(
            arrays['b_tmp_%s' % target.name],
            b,
            subdomain=eq.subdomain
        ))

    # TODO: Add corresponding submatrix as an argument
    return FieldData(
        target=target,
        matvecs=matvecs,
        formfuncs=formfuncs,
        formrhs=formrhs,
        arrays=arrays,
        dmda=dmda
    )

def separate_eqn(eqn, target):
    """
    Separate the equation into two separate expressions,
    where F(target) = b.
    """
    zeroed_eqn = Eq(eqn.lhs - eqn.rhs, 0)
    zeroed_eqn = eval_time_derivatives(zeroed_eqn.lhs)
    target_funcs = set(generate_targets(zeroed_eqn, target))
    b, F_target = remove_targets(zeroed_eqn, target_funcs)
    return -b, F_target, target_funcs


def generate_targets(eq, target):
    """
    Extract all the functions that share the same time index as the target
    but may have different spatial indices.
    """
    funcs = retrieve_functions(eq)
    if isinstance(target, TimeFunction):
        time_idx = target.indices[target.time_dim]
        targets = [
            f for f in funcs if f.function is target.function and time_idx
            in f.indices
        ]
    else:
        targets = [f for f in funcs if f.function is target.function]
    return targets


def targets_to_arrays(array, targets):
    """
    Map each target in `targets` to a corresponding array generated from `array`,
    matching the spatial indices of the target.

    Example:
    --------
    >>> array
    vec_u(x, y)

    >>> targets
    {u(t + dt, x + h_x, y), u(t + dt, x - h_x, y), u(t + dt, x, y)}

    >>> targets_to_arrays(array, targets)
    {u(t + dt, x - h_x, y): vec_u(x - h_x, y),
     u(t + dt, x + h_x, y): vec_u(x + h_x, y),
     u(t + dt, x, y): vec_u(x, y)}
    """
    space_indices = [
        tuple(f.indices[d] for d in f.space_dimensions) for f in targets
    ]
    array_targets = [
        array.subs(dict(zip(array.indices, i))) for i in space_indices
    ]
    return dict(zip(targets, array_targets))


@singledispatch
def remove_targets(expr, targets):
    return (0, expr) if expr in targets else (expr, 0)


@remove_targets.register(sympy.Add)
def _(expr, targets):
    if not any(expr.has(t) for t in targets):
        return (expr, 0)

    args_b, args_F = zip(*(remove_targets(a, targets) for a in expr.args))
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_targets.register(Mul)
def _(expr, targets):
    if not any(expr.has(t) for t in targets):
        return (expr, 0)

    args_b, args_F = zip(*[remove_targets(a, targets) if any(a.has(t) for t in targets)
                           else (a, a) for a in expr.args])
    return (expr.func(*args_b, evaluate=False), expr.func(*args_F, evaluate=False))


@remove_targets.register(Derivative)
def _(expr, targets):
    return (0, expr) if any(expr.has(t) for t in targets) else (expr, 0)


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
    the `LinearSolver` object, which is passed through the initial lowering
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
    return dict(zip(time_indices, tau_symbs))
