import numpy as np
from devito import Grid, Function, Eq, Operator, SubDomain
from devito import configuration
from devito.types import PETScSolve
configuration['opt'] = 'noop'


# WORKING BUT NEED TO ADJUST LOOPS 

# solve laplace equation with Dirichlet BCs AND NEUMANN BCs.

# # no of grid nodes in each direction
nx = 13
ny = 7
Lx = np.float64(2.0)
Ly = np.float64(1.0)

# from IPython import embed; embed()
dx = Lx / np.float64(nx - 1)
dy = Ly / np.float64(ny - 1)

s_o = 2

# this will be Neumann subdomain
class SubTop(SubDomain):
    name = 'subtop'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('right', self.S_O//2)}
sub1 = SubTop(s_o)

# this will be Neumann subdomain
class SubBottom(SubDomain):
    name = 'subbottom'
    def __init__(self, S_O):
        super().__init__()
        self.S_O = S_O

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('left', self.S_O//2)}
sub2 = SubBottom(s_o)

# this will be dirichlet subdomain
class SubLeft(SubDomain):
    name = 'subleft'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: y}
sub3 = SubLeft()

# this will be dirichlet subdomain
class SubRight(SubDomain):
    name = 'subright'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: y}
sub4 = SubRight()

from devito import sign, norm
from devito.symbolics import retrieve_functions, INT

def neumann_bottom(eq, subdomain):
    """
    Modify a stencil such that it is folded back on
    itself rather than leaving the model domain. This is
    achieved by replacing the symbolic indices for some
    function of those indices. Depending on how this is
    done, this can be used to implement a pressure or
    velocity free-surface. This is the MPI-safe method
    of implementing a free-surface boundary condition
    in Devito.
    
    Parameters
    ----------
    eq : Eq
        The update stencil to modify
    subdomain : FreeSurface
        The subdomain in which the modification is
        applied
    """
    lhs, rhs = eq.evaluate.args
    
    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent
    
    # Functions present in stencil
    funcs = retrieve_functions(rhs)
    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] < 0:
            # If point position in stencil is negative
            # Substitute the dimension for its subdomain companion
            # Get the symbolic sign of this
            s = sign(yind.subs({y: yfs, y.spacing: 1}))
            # Symmetric mirror
            # Substitute where index is negative for +ve where index is positive
            mapper.update({f: f.subs({yind: INT(abs(yind))})})

    return Eq(lhs, rhs.subs(mapper), subdomain=subdomain)


def neumann_top(eq, subdomain):
    """
    Modify a stencil such that it is folded back on
    itself rather than leaving the model domain. This is
    achieved by replacing the symbolic indices for some
    function of those indices. Depending on how this is
    done, this can be used to implement a pressure or
    velocity free-surface. This is the MPI-safe method
    of implementing a free-surface boundary condition
    in Devito.
    
    Parameters
    ----------
    eq : Eq
        The update stencil to modify
    subdomain : FreeSurface
        The subdomain in which the modification is
        applied
    """
    lhs, rhs = eq.evaluate.args
    
    # Get vertical subdimension and its parent
    yfs = subdomain.dimensions[-1]
    y = yfs.parent
    
    # Functions present in stencil
    funcs = retrieve_functions(rhs)
    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind - y).as_coeff_Mul()[0] > 0:
            # If point position in stencil is negative
            # Substitute the dimension for its subdomain companion
            # Get the symbolic sign of this
            s = sign(yind.subs({y: yfs, y.spacing: 1}))
            # Symmetric mirror
            # Substitute where index is negative for +ve where index is positive
            mapper.update({f: f.subs({yind: INT(abs(yind))})})

    return Eq(lhs, rhs.subs(mapper), subdomain=subdomain)


# from IPython import embed; embed()

grid = Grid(shape=(nx, ny), extent=(Lx, Ly), subdomains=(sub1,sub2,sub3,sub4,), dtype=np.float64)

pn = Function(name='pn', grid=grid, space_order=2, dtype=np.float64)

rhs = Function(name='rhs', grid=grid, space_order=2, dtype=np.float64)


eqn = Eq(rhs, pn.laplace, subdomain=grid.subdomains['interior'])

dirBCs = Function(name='dirBCs', grid=grid, space_order=2, dtype=np.float64)

tmp = np.linspace(0, Ly, ny).astype(np.float64)*np.float64(np.pi)
right_val = np.float64(np.cos(tmp))
dirBCs.data[-1, :] = right_val

# make sure field you're solving for satsifys BCs
pn.data[-1, :] = right_val
# pn.data[:] = np.arange(1, 92).reshape(nx, ny)
# print(pn.data)
rhs.data[:] = np.float64(0.)

# from IPython import embed; embed()
bc_top = neumann_top(eqn, sub1)
bc_bottom = neumann_bottom(eqn, sub2)
bc_left = Eq(pn, dirBCs, subdomain=sub3)
bc_right = Eq(pn, dirBCs, subdomain=sub4)

bcs = [bc_top] + [bc_bottom] + [bc_left] + [bc_right]
petsc = PETScSolve(eqn, pn, bcs=bcs, solver_parameters={'ksp_type': 'gmres',
                                                                'pc_type': 'jacobi',
                                                                'ksp_rtol': 1e-7,
                                                                'ksp_max_it': 10000})

# op = Operator([eqn] + [bc_top] + [bc_bottom] + [bc_left] + [bc_right], opt='noop')
op = Operator(petsc, opt='noop')
print(op.arguments())
op.apply()
# print(pn.data[:])
print(op.ccode)
# print(pn.data[:])

import pandas as pd
pd.DataFrame(pn.data[:]).to_csv("1.csv", header=None, index=None)