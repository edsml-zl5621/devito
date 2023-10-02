from devito import *
import numpy as np
configuration['opt'] = 'noop'


# solve laplace equation with Dirichlet BCs.
# All bcs are zero apart from top boundary

# no of grid nodes in each direction
nx = 5
ny = 5
Lx = 1.
Ly = 1.
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

grid = Grid(shape=(nx, ny), extent=(Lx, Ly))

pn = Function(name='pn', grid=grid, space_order=2)
p = Function(name='p', dimensions=grid.subdomains['interior'].dimensions,
             shape=grid.subdomains['interior'].shape, space_order=0)

eqn = Eq(p, pn.laplace, subdomain=grid.subdomains['interior'])

p.data[:] = 0.
pn.data[:] = 0.
pn.data[:, -1] = np.sin(np.linspace(0, Lx, nx)*np.pi/Lx)

x, y = grid.dimensions

# Create boundary condition expressions
# Required in this form for mat-free code
bc = [Eq(p[0, y], pn[0, y])] 
bc += [Eq(p[nx-1, y], pn[nx-1, y])]  
bc += [Eq(p[x, 0], pn[x, 0])]  
bc += [Eq(p[x, ny-1], pn[x, ny-1])] 

# build the op
op = Operator(expressions= [eqn] + bc)

op.apply()

print(op.ccode)
print(pn.data[:])

