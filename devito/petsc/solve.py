from functools import singledispatch

import sympy

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.derivative import Derivative
from devito.types import Eq, Symbol, SteppingDimension, TimeFunction
from devito.types.equation import InjectSolveEq
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions
from devito.tools import as_tuple, filter_ordered
from devito.petsc.types import LinearSolveExpr, PETScArray, DMDALocalInfo, FieldData, MultipleFieldData


__all__ = ['PETScSolve', 'EssentialBC']


def PETScSolve(eqns_targets, target=None, solver_parameters=None, **kwargs):
    if target is not None:
        injectsolve = InjectSolve().build({target: eqns_targets}, solver_parameters)
        # TODO: the return should probs just be the InjectSolveEq object, not put into a list
        return injectsolve

    # If users want segregated solvers, they create multiple PETScSolve objects,
    # rather than passing multiple targets to a single PETScSolve object
    else:
        injectsolve = InjectSolveNested().build(eqns_targets, solver_parameters)
        return injectsolve


class InjectSolve:
    def build(self, eqns_targets, solver_parameters):
        target, funcs, fielddata, time_mapper = self.build_eq(
            eqns_targets, solver_parameters
        )
        # Placeholder equation for inserting calls to the solver and generating
        # correct time loop etc.
        return [InjectSolveEq(target, LinearSolveExpr(
                funcs, solver_parameters, fielddata=fielddata, time_mapper=time_mapper,
                localinfo=localinfo))]

    def build_eq(self, eqns_targets, solver_parameters):
        target, eqns = next(iter(eqns_targets.items()))
        eqns = as_tuple(eqns)
        funcs = get_funcs(eqns)
        time_mapper = generate_time_mapper(funcs)
        fielddata = self.generate_field_data(eqns, target, time_mapper)
        return target, tuple(funcs), fielddata, time_mapper

    def generate_field_data(self, eqns, target, time_mapper):
        # TODO: change these names
        # TODO: fielddata needs to extend/change to be a data per submatrix, i.e a list of matvecs, formfuncs, formrhs etc for each submatrix with a label /indexsert identifying it's location in the larger Jacobian
        prefixes = ['y_matvec', 'x_matvec', 'f_formfunc', 'x_formfunc', 'b_tmp']

        arrays = {
            p: PETScArray(name='%s_%s' % (p, target.name),
                        target=target,
                        liveness='eager',
                        localinfo=localinfo)
            for p in prefixes
        }

        matvecs, formfuncs, formrhs = zip(
            *[self.build_callback_eqns(eq, target, arrays, time_mapper) for eq in eqns]
        )

        # todo, I think the prefixes could be specific to the solve not the fielddata ?
        tmp =  FieldData(
            target=target,
            matvecs=matvecs,
            formfuncs=formfuncs,
            formrhs=formrhs,
            arrays=arrays,
        )
        return tmp

    def build_callback_eqns(self, eq, target, arrays, time_mapper):
        b, F_target, targets = separate_eqn(eq, target)
        name = target.name

        matvec = self.make_matvec(eq, F_target, arrays, name, targets)
        formfunc = self.make_formfunc(eq, F_target, arrays, name, targets)
        formrhs = self.make_rhs(eq, b, arrays, name)

        return tuple(expr.subs(time_mapper) for expr in (matvec, formfunc, formrhs))

    def make_matvec(self, eq, F_target, arrays, name, targets):
        if isinstance(eq, EssentialBC):
            return Eq(
                arrays['y_matvec'], arrays['x_matvec'],
                subdomain=eq.subdomain
            )
        else:
            return Eq(
                arrays['y_matvec'],
                F_target.subs(targets_to_arrays(arrays['x_matvec'], targets)),
                subdomain=eq.subdomain
            )

    def make_formfunc(self, eq, F_target, arrays, name, targets):
        if isinstance(eq, EssentialBC):
            return Eq(
                arrays['f_formfunc'], 0.,
                subdomain=eq.subdomain
            )
        else:
            return Eq(
                arrays['f_formfunc'],
                F_target.subs(targets_to_arrays(arrays['x_formfunc'], targets)),
                subdomain=eq.subdomain
            )

    def make_rhs(self, eq, b, arrays, name):
        if isinstance(eq, EssentialBC):
            return Eq(
                arrays['b_tmp'], 0,
                subdomain=eq.subdomain
            )
        else:
            return Eq(
                arrays['b_tmp'], b,
                subdomain=eq.subdomain
            )


class InjectSolveNested(InjectSolve):
    def build_eq(self, eqns_targets, solver_parameters):
        combined_eqns = [item for sublist in eqns_targets.values() for item in sublist]
        funcs = get_funcs(combined_eqns)
        time_mapper = generate_time_mapper(funcs)

        all_data = MultipleFieldData()

        for target, eqns in eqns_targets.items():
            eqns = as_tuple(eqns)
            fielddata = self.generate_field_data(eqns, target, time_mapper)
            all_data.add_field_data(fielddata)

        return target, tuple(funcs), all_data, time_mapper


class EssentialBC(Eq):
    pass


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
    present in the initial lowering.
    NOTE: All functions used in PETSc callback functions are attached to
    the `LinearSolveExpr` object, which is passed through the initial lowering
    (and subsequently dropped and replaced with calls to run the solver).
    Therefore, the appropriate time loop will always be correctly generated inside
    the main kernel.

    Examples
    --------
    >>> funcs = [
    >>>     f1(t + dt, x, y),
    >>>     g1(t + dt, x, y),
    >>>     g2(t, x, y),
    >>>     f1(t, x, y)
    >>> ]
    >>> generate_time_mapper(funcs)
    {t + dt: tau0, t: tau1}

    """
    time_indices = list({
        i if isinstance(d, SteppingDimension) else d
        for f in funcs
        for i, d in zip(f.indices, f.dimensions)
        if d.is_Time
    })
    tau_symbs = [Symbol('tau%d' % i) for i in range(len(time_indices))]
    return dict(zip(time_indices, tau_symbs))


def get_funcs(eqns):
    funcs = [
        func
        for eq in eqns
        for func in retrieve_functions(eval_time_derivatives(eq.lhs - eq.rhs))
    ]
    return filter_ordered(funcs)


lhs = Symbol('lhs')
localinfo = DMDALocalInfo(name='info', liveness='eager')
