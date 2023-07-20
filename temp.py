import pytest

from ctypes import c_void_p
import cgen
import sympy


from devito import (Eq, Grid, Function, TimeFunction, Operator, Dimension,  # noqa
                    switchconfig, dimensions, SpaceDimension)
from devito.ir.iet import (Call, Callable, Conditional, DummyExpr, Iteration, List,
                           Lambda, ElementalFunction, CGen, FindSymbols,
                           filter_iterations, make_efunc, retrieve_iteration_tree,
                           Definition, Expression, Transformer, FindNodes)
from devito.ir import SymbolRegistry
from devito.passes.iet.engine import Graph
from devito.passes.iet.languages.C import CDataManager
from devito.symbolics import Byref, FieldFromComposite, InlineIf, Macro
from devito.tools import as_tuple
from devito.types import Array, LocalObject, Symbol
from devito.passes.iet.petsc import PetscObject
from devito.ir.equations import DummyEq

# dims_op = {'x': SpaceDimension(name='x'),
#             'y': SpaceDimension(name='y')}

# grid = Grid(shape=(5, 5))
# symbs_op = {'u': Function(name='u', grid=grid).indexify()}

# def get_exprs(u):
#     return [Expression(DummyEq(u, u+1))]

# exprs_op = get_exprs(symbs_op['u'])

# def get_iters(dims_op):
#     return [lambda ex: Iteration(ex, dims_op['x'], (0, 4, 1)),
#             lambda ex: Iteration(ex, dims_op['y'], (0, 4, 1))]

# iters_op = get_iters(dims_op)

# def get_block1(exprs_op, iters_op):
#     return iters_op[0](iters_op[1](exprs_op[0]))

# block1 = get_block1(exprs_op, iters_op)


# u = TimeFunction(name='u', grid=grid)
# iet = Callable('kernel', block1, 'int', ())

# # MyMatMultShell = Call()
# MyMatMultShell = Callable('MyMatMultShell', block1, 'void', ())
# # from IPython import embed; embed()
# iet = Transformer({iet.body: Call(MyMatMultShell.name)}, nested=True).visit(iet)

# # from IPython import embed; embed()


# tmp = FindNodes(Call).visit(iet)
# from IPython import embed; embed()
# # print(iet.ccode)






dims_op = {'x': SpaceDimension(name='x'),
            'y': SpaceDimension(name='y')}

grid = Grid(shape=(5, 5))
symbs_op = {'u': Function(name='u', grid=grid).indexify()}

def get_exprs(u):
    return [Expression(DummyEq(u, u+1))]

exprs_op = get_exprs(symbs_op['u'])

def get_iters(dims_op):
    return [lambda ex: Iteration(ex, dims_op['x'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims_op['y'], (0, 4, 1))]

iters_op = get_iters(dims_op)

def get_block1(exprs_op, iters_op):
    return iters_op[0](iters_op[1](exprs_op[0]))

block1 = get_block1(exprs_op, iters_op)

MyMatMultShell = Callable('MyMatMultShell', block1, 'void', ())

def get_block2(exprs_op, iters_op):
    return iters_op[0](iters_op[1]([exprs_op[0], Call(MyMatMultShell.name)]))

block2 = get_block2(exprs_op, iters_op)

# u = TimeFunction(name='u', grid=grid)
iet = Callable('kernel', block2, 'int', ())


# print(iet.ccode)
# from IPython import embed; embed()

n_s = []
for tree in retrieve_iteration_tree(iet):
    for i in reversed(tree):
        for n in tree[:tree.index(i)+1]:
            n_s.append(n)

for i in n_s:
    print(i.properties)

# mapper = {}
# for tree in retrieve_iteration_tree(iet):
#     for i in reversed(tree):
#         if i in mapper:
#             # Already seen this subtree, skip
#             break
#         if FindNodes(Call).visit(i):
#             print("hi")
#             mapper.update({n: n._rebuild(properties=set(n.properties))
#                             for n in tree[:tree.index(i)+1]})
#             break

# print(mapper)
# iet = Transformer(mapper, nested=True).visit(iet)

# print(iet.ccode)