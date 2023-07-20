# from devito import *
# grid = Grid(shape=(5, 5), extent=(1., 1.))
# p = Function(name='p', grid=grid, space_order=2)
# pn = Function(name='pn', grid=grid, space_order=2)
# op = Operator(Eq(p, pn.laplace))

# print(op.ccode)


from mpi4py import MPI
print(f"Hi, I'm rank %d." % MPI.COMM_WORLD.rank)
