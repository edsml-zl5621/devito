from functools import singledispatch

import cgen
import sympy
from ctypes import c_void_p

from devito.finite_differences import Max, Min
from devito.ir import (Any, Forward, Iteration, List, Prodder, FindApplications,
                       FindNodes, Transformer, Uxreplace, filter_iterations,
                       retrieve_iteration_tree, MapExprStmts, FindSymbols, EntryFunction)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import evalrel, has_integer_args
from devito.tools import as_mapper, split
from devito import (Grid, Function)
from devito.ir.iet import (Callable, DummyExpr, Block)
from devito.tools import as_mapper, as_list, as_tuple, filter_sorted, flatten
from operator import itemgetter
from collections import OrderedDict


from devito import Grid, Function
from devito.ir.iet import (Call, Callable, Conditional, DummyExpr, Iteration, List,
                           Lambda, ElementalFunction, CGen, FindSymbols,
                           filter_iterations, make_efunc, retrieve_iteration_tree,
                           Definition, Expression, Transformer)
from devito.ir import SymbolRegistry
from devito.passes.iet.engine import Graph
from devito.passes.iet.languages.C import CDataManager
from devito.symbolics import Byref, FieldFromComposite, InlineIf, Macro
from devito.tools import as_tuple
from devito.types import Array, LocalObject, Symbol
# from devito.passes.iet.petsc import PetscObject
from devito.ir.equations import DummyEq

__all__ = ['avoid_denormals', 'hoist_prodders', 'relax_incr_dimensions',
           'generate_macros', 'minimize_symbols', 'lower_petsc']


@iet_pass
def avoid_denormals(iet, platform=None):
    """
    Introduce nodes in the Iteration/Expression tree that will expand to C
    macros telling the CPU to flush denormal numbers in hardware. Denormals
    are normally flushed when using SSE-based instruction sets, except when
    compiling shared objects.
    """
    # There is unfortunately no known portable way of flushing denormal to zero.
    # See for example: https://stackoverflow.com/questions/59546406/\
    #                       a-robust-portable-way-to-set-flush-denormals-to-zero
    try:
        if 'sse' not in platform.known_isas:
            return iet, {}
    except AttributeError:
        return iet, {}

    if iet.is_ElementalFunction:
        return iet, {}

    header = (cgen.Comment('Flush denormal numbers to zero in hardware'),
              cgen.Statement('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)'),
              cgen.Statement('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)'),
              cgen.Line())

    body = iet.body._rebuild(body=(List(header=header),) + iet.body.body)
    iet = iet._rebuild(body=body)

    return iet, {'includes': ('xmmintrin.h', 'pmmintrin.h')}


@iet_pass
def hoist_prodders(iet):
    """
    Move Prodders within the outer levels of an Iteration tree.
    """
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        for prodder in FindNodes(Prodder).visit(tree.root):
            if prodder._periodic:
                try:
                    key = lambda i: i.dim.is_Block and i.dim.step != 1
                    candidate = filter_iterations(tree, key)[-1]
                except IndexError:
                    # Fallback: use the outermost Iteration
                    candidate = tree.root
                mapper[candidate] = candidate._rebuild(nodes=(candidate.nodes +
                                                              (prodder._rebuild(),)))
                mapper[prodder] = None

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}


@iet_pass
def relax_incr_dimensions(iet, options=None, **kwargs):
    """
    This pass adjusts the bounds of blocked Iterations in order to include the "remainder
    regions".  Without the relaxation that occurs in this pass, the only way to iterate
    over the entire iteration space is to have step increments that are perfect divisors
    of the iteration space (e.g. in case of an iteration space of size 67 and block size
    8 only 64 iterations would be computed, as `67 - 67mod8 = 64`.

    A simple 1D example: nested Iterations are transformed from:

    <Iteration x0_blk0; (x_m, x_M, x0_blk0_size)>
        <Iteration x; (x0_blk0, x0_blk0 + x0_blk0_size - 1, 1)>

    to:

    <Iteration x0_blk0; (x_m, x_M, x0_blk0_size)>
        <Iteration x; (x0_blk0, Min(x_M, x0_blk0 + x0_blk0_size - 1)), 1)>

    """
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree if i.dim.is_Block]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        assert all(i.direction in (Forward, Any) for i in iterations)
        outer, inner = split(iterations, lambda i: not i.dim.parent.is_Block)

        # Get root's `symbolic_max` out of each outer Dimension
        roots_max = {i.dim.root: i.symbolic_max for i in outer}

        # Process inner iterations and adjust their bounds
        for n, i in enumerate(inner):
            # If definitely in-bounds, as ensured by a prior compiler pass, then
            # we can skip this step
            if i.is_Inbound:
                continue

            # The Iteration's maximum is the Min of (a) the `symbolic_max` of current
            # Iteration e.g. `x0_blk0 + x0_blk0_size - 1` and (b) the `symbolic_max`
            # of the current Iteration's root Dimension e.g. `x_M`. The generated
            # maximum will be `Min(x0_blk0 + x0_blk0_size - 1, x_M)

            # In some corner cases an offset may be added (e.g. after CIRE passes)
            # E.g. assume `i.symbolic_max = x0_blk0 + x0_blk0_size + 1` and
            # `i.dim.symbolic_max = x0_blk0 + x0_blk0_size - 1` then the generated
            # maximum will be `Min(x0_blk0 + x0_blk0_size + 1, x_M + 2)`

            root_max = roots_max[i.dim.root] + i.symbolic_max - i.dim.symbolic_max
            iter_max = evalrel(min, [i.symbolic_max, root_max])
            mapper[i] = i._rebuild(limits=(i.symbolic_min, iter_max, i.step))

    if mapper:
        iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}


# @iet_pass
# def generate_macros(iet):
#     applications = FindApplications().visit(iet)
#     headers = set().union(*[_generate_macros(i) for i in applications])
#     # from IPython import embed; embed()
#     return iet, {'headers': headers}


# playing around with generate_macros
# this is how you would add an extra header that is not default?
@iet_pass
def generate_macros(iet):
    # return iet, {'headers': {('MAX(a,b)', ('(((a) > (b)) ? (a) : (b))'))}}
    return iet, {'headers': {('PI', ('3.14'))}}


@singledispatch
def _generate_macros(expr):
    return set()


@_generate_macros.register(Min)
@_generate_macros.register(sympy.Min)
def _(expr):
    if has_integer_args(*expr.args) and len(expr.args) == 2:
        return {('MIN(a,b)', ('(((a) < (b)) ? (a) : (b))'))}
    else:
        return set()


@_generate_macros.register(Max)
@_generate_macros.register(sympy.Max)
def _(expr):
    if has_integer_args(*expr.args) and len(expr.args) == 2:
        return {('MAX(a,b)', ('(((a) > (b)) ? (a) : (b))'))}
    else:
        return set()


@iet_pass
def minimize_symbols(iet):
    """
    Remove unneccesary symbols. Currently applied sub-passes:

        * Remove redundant ModuloDimensions (e.g., due to using the
          `save=Buffer(2)` API)
    """
    iet = remove_redundant_moddims(iet)

    return iet, {}


def remove_redundant_moddims(iet):
    key = lambda d: d.is_Modulo and d.origin is not None
    mds = [d for d in FindSymbols('dimensions').visit(iet) if key(d)]
    if not mds:
        return iet

    mapper = as_mapper(mds, key=lambda md: md.origin % md.modulo)

    subs = {}
    for k, v in mapper.items():
        chosen = v.pop(0)
        subs.update({d: chosen for d in v})

    body = Uxreplace(subs).visit(iet.body)
    iet = iet._rebuild(body=body)

    # ModuloDimensions are defined in Iteration headers, hence they must be
    # removed from there too
    subs = {}
    for n in FindNodes(Iteration).visit(iet):
        if not set(n.uindices) & set(mds):
            continue
        subs[n] = n._rebuild(uindices=filter_ordered(n.uindices))

    iet = Transformer(subs, nested=True).visit(iet)

    return iet




@iet_pass
def lower_petsc(iet):
    call_back = Callable('call_back', iet.body, 'void', parameters=iet.parameters)
    iet = Transformer({iet.body: Call(call_back.name)}).visit(iet)
    return iet, {'efuncs': [call_back]}


