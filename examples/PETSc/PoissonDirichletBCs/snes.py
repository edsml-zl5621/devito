from devito import *
from devito.types import PETScSolve
import pandas as pd
from devito import configuration
import numpy as np
configuration['opt'] = 'noop'

# Solving pn.laplace = 2x(y - 1)(y - 2x + xy + 2)e^(x-y)
# Constant zero Dirichlet BCs.

nx = 11
ny = 11

Lx = np.float64(1.)
Ly = np.float64(1.)

dx = Lx / np.float64(nx - 1)
dy = Ly / np.float64(ny - 1)

# Subdomains to implement BCs
class SubTop(SubDomain):
    name = 'subtop'
    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('right', 1)}
sub1 = SubTop()

class SubBottom(SubDomain):
    name = 'subbottom'
    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('left', 1)}
sub2 = SubBottom()

class SubLeft(SubDomain):
    name = 'subleft'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 1), y: y}
sub3 = SubLeft()

class SubRight(SubDomain):
    name = 'subright'
    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 1), y: y}
sub4 = SubRight()

grid = Grid(shape=(nx, ny), extent=(Lx, Ly), subdomains=(sub1,sub2,sub3,sub4,), dtype=np.float64)

pn = Function(name='pn', grid=grid, space_order=2, dtype=np.float64)
rhs = Function(name='rhs', grid=grid, space_order=2, dtype=np.float64)
eqn = Eq(rhs, pn.laplace, subdomain=grid.interior)
# Initial guess - satisfies BCs
pn.data[:] = np.float64(0.)
tmpx = np.linspace(0, Lx, nx).astype(np.float64)
tmpy = np.linspace(0, Ly, ny).astype(np.float64)

Y, X = np.meshgrid(tmpx, tmpy)

rhs.data[:] = np.float64(2.0*X*(Y-1.0)*(Y - 2.0*X + X*Y + 2.0)) * np.float64(np.exp(X-Y))
# # Create boundary condition expressions using subdomains
x, y = grid.dimensions

boundaries = Function(name='boundaries', grid=grid, space_order=2, dtype=np.float64)

boundaries.data[:] = 0.
bcs = [Eq(pn, boundaries, subdomain=sub1)]
bcs += [Eq(pn, boundaries, subdomain=sub2)]
bcs += [Eq(pn, boundaries, subdomain=sub3)]
bcs += [Eq(pn, boundaries, subdomain=sub4)]

# # ksp type, pc type and relative tolerance.
petsc = PETScSolve(eqn, pn, bcs=bcs, solver_parameters={'ksp_type': 'gmres',
                                                                'pc_type': 'jacobi',
                                                                'ksp_rtol': 1e-7,
                                                                'ksp_max_it': 10000})
# Build the op
op = Operator(petsc, opt='noop')
op.apply()
print(op.ccode)
print(op.arguments())