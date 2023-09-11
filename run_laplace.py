from devito import *
import numpy as np
configuration['opt'] = 'noop'

nx = 5
ny = 5
Lx = 1.
Ly = 1.
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

grid = Grid(shape=(nx, ny), extent=(Lx, Ly))

pn = Function(name='pn', grid=grid, space_order=2)
p = Function(name='p', dimensions=grid.subdomains['interior'].dimensions, shape=grid.subdomains['interior'].shape, space_order=0)

eqn = Eq(p, pn.laplace, subdomain=grid.subdomains['interior'])

p.data[:] = 0.
pn.data[:] = 0.
pn.data[:, -1] = np.sin(np.linspace(0, Lx, nx)*np.pi/Lx)

x, y = grid.dimensions
bc_top = Function(name='bc_top', shape=(nx, ), dimensions=(x, ))
bc_top.data[:] = np.linspace(0, Lx, nx)
bc_top.data[:] = np.sin(bc_top.data[:]*np.pi/Lx)

# Create boundary condition expressions
bc = [Eq(pn[0, y], 0.)]  # pn = 0 @ x = 0
bc += [Eq(pn[nx-1, y], 0.)]  # pn = 0 @ x = Lx
bc += [Eq(pn[x, 0], 0.)]  # pn = 0 @ y = 0
bc += [Eq(pn[x, ny-1], bc_top[x])]  # pn = sin(pi*x/Lx) @ y = Ly

# build the op
op = Operator(expressions= [eqn] + bc)

# op = Operator(eqn)


op.apply()

print(op.ccode)
print(pn.data[:])
# print(op.arguments())
# # print(p.laplace.evaluate)


for_file = pn.data[:]

# from IPython import embed; embed()

# for_file = for_file.reshape((pn.shape[0])**2)

import pandas as pd

pd.DataFrame(for_file).to_csv("run_laplace_results/1.csv", header=None, index=None)