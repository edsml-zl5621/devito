from devito import *


grid = Grid(shape=(5, 5), extent=(1., 1.))
p = Function(name='p', grid=grid, space_order=2)
pn = Function(name='pn', grid=grid, space_order=2)
op = Operator(Eq(p, pn.laplace))


op.apply()


# print(op.ccode)