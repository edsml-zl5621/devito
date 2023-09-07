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
# p = Function(name='p', grid=grid, space_order=2)

# pn = Function(name='pn', grid=grid, space_order=2)
# print(pn.laplace.evaluate)

# xi, yi = grid.subdomains['interior'].dimensions


# from IPython import embed; embed()
pn = Function(name='pn', grid=grid, space_order=8)
# p = Function(name='p', grid=grid, space_order=2)

# p = Function(name='p', dimensions=(x_p, y_p), shape=grid.subdomains['interior'].shape, space_order=0)
p = Function(name='p', dimensions=grid.subdomains['interior'].dimensions, shape=grid.subdomains['interior'].shape, space_order=0)

# p = Function(name='p', grid=grid, space_order=0)

# from IPython import embed; embed()
eqn = Eq(p, pn.laplace, subdomain=grid.subdomains['interior'])
# eqn = Eq(p, pn.laplace)
# eqn = Eq(p, pn.laplace, subdomain=grid.subdomains['interior'])

# from IPython import embed; embed()
pn.data[:] = 0.
pn.data[:, -1] = np.sin(np.linspace(0, Lx, nx)*np.pi/Lx)

# pn.data[:] = 0.

# x, y = grid.dimensions
# bc_top = Function(name='bc_top', shape=(nx, ), dimensions=(x, ))
# bc_top.data[:] = np.linspace(0, Lx, nx)
# bc_top.data[:] = np.sin(bc_top.data[:]*np.pi/Lx)
# # print(bc_top.data[:])

# # Create boundary condition expressions
# bc = [Eq(p[0, y], 0.)]  # p = 0 @ x = 0
# bc += [Eq(p[nx-1, y], 0.)]  # p = 0 @ x = Lx
# bc += [Eq(p[x, 0], 0.)]  # p = 0 @ y = 0
# bc += [Eq(p[x, ny-1], bc_top[x])]  # p = sin(pi*x/Lx) @ y = Ly

# # Now we can build the operator that we need
# op = Operator(expressions= [eqn] + bc)

# print(p.data[:])

op = Operator(eqn)

# from IPython import embed; embed()

op.apply()

print(op.ccode)
print(pn.data[:])
# print(op.arguments())
# # print(p.laplace.evaluate)