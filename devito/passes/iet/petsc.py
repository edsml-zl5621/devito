from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (FindNodes, Call, MatVecAction, RHSLinearSystem,
                           Transformer, FindSymbols, LinearSolverExpression,
                           MapNodes, Iteration, Callable, Callback)
from devito.types import (PetscMPIInt, PETScStruct, DMDALocalInfo, DM, Mat,
                          Vec, KSP, PC, SNES, PetscErrorCode)
from devito.symbolics import Byref, Macro
import cgen as c

__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # Check if PETScSolve was used.
    petsc_nodes = FindNodes(MatVecAction).visit(iet)

    if not petsc_nodes:
        return iet, {}

    else:
        # Collect all petsc solution fields
        unique_targets = list(set([i.expr.rhs.target for i in petsc_nodes]))

        # Initalize PETSc
        init = init_petsc(**kwargs)

        # Create context data struct
        struct = build_struct(iet)

        objs = build_core_objects(unique_targets[-1], **kwargs)

        # Create core PETSc calls (not specific to each PETScSolve)
        core = core_petsc(unique_targets[-1], struct, objs, **kwargs)

        # from IPython import embed; embed()
        matvec_mapper = MapNodes(Iteration, MatVecAction, 'groupby').visit(iet)

        main_mapper = {}

        setup = []
        efuncs = []
        for target in unique_targets:

            solver_objs = build_solver_objs(target)

            matvec_callback_body_iters = []

            solver_setup = False

            for iter, (matvec,) in matvec_mapper.items():

                if matvec.expr.rhs.target == target:
                    if not solver_setup:
                        solver = generate_solver_calls(solver_objs, objs, matvec)
                        setup.extend(solver)
                        solver_setup = True

                    matvec_callback_body_iters.append(iter[0])
                    main_mapper.update({iter[0]: None})

            matvec_callback, matvec_op = create_matvec_callback(
                target, matvec_callback_body_iters, solver_objs, objs)

            setup.append(matvec_op)
            setup.append(c.Line())

            efuncs.append(matvec_callback)

        
        # from IPython import embed; embed()
        iet = Transformer(main_mapper).visit(iet)

        # Remove the LinSolveExpr that was utilised above to carry metadata
        lin_solver_mapper = {expr:
                  expr._rebuild(expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr))
                  for expr in FindNodes(LinearSolverExpression).visit(iet)}

        iet = Transformer(lin_solver_mapper).visit(iet)

        body = iet.body._rebuild(init=init, body=core + tuple(setup) + iet.body.body)
        iet = iet._rebuild(body=body)

        return iet, {'efuncs': efuncs}


def init_petsc(**kwargs):

    # Initialize PETSc -> for now, assuming all solver options have to be
    # specifed via the parameters dict in PETScSolve
    # TODO: Are users going to be able to use PETSc command line arguments?
    # In firedrake, they have an options_prefix for each solver, enabling the use
    # of command line options
    initialize = Call('PetscCall', [Call('PetscInitialize',
                                         arguments=[Null, Null,
                                                    Null, Null])])

    return tuple([initialize])


def build_struct(iet):
    # Place all context data required by the shell routines
    # into a PETScStruct
    usr_ctx = []

    basics = FindSymbols('basics').visit(iet)
    avoid = FindSymbols('dimensions|indexedbases').visit(iet)
    usr_ctx.extend(data for data in basics if data not in avoid)

    return PETScStruct('ctx', usr_ctx)


def core_petsc(target, struct, objs, **kwargs):

    # MPI
    call_mpi = Call('PetscCallMPI', [Call('MPI_Comm_size',
                                          arguments=[objs['comm'],
                                                     Byref(objs['size'])])])

    # Create DMDA
    dmda = create_dmda(target, objs)
    dm_setup = Call('PetscCall', [Call('DMSetUp', arguments=[objs['da']])])
    dm_app_ctx = Call('PetscCall', [Call('DMSetApplicationContext',
                                         arguments=[objs['da'], struct])])
    dm_mat_type = Call('PetscCall', [Call('DMSetMatType',
                                          arguments=[objs['da'], 'MATSHELL'])])
    dm_local_info = Call('PetscCall', [Call('DMDAGetLocalInfo',
                                            arguments=[objs['da'], Byref(objs['info'])])])

    return tuple([call_mpi, dmda, dm_setup, dm_app_ctx, dm_mat_type, dm_local_info, c.Line()])


def build_core_objects(target, **kwargs):

    if kwargs['options']['mpi']:
        communicator = target.grid.distributor._obj_comm
    else:
        communicator = 'PETSC_COMM_SELF'

    return {'da': DM(name='da'),
            'size': PetscMPIInt(name='size'),
            'info': DMDALocalInfo(name='info'),
            'comm': communicator,
            'err': PetscErrorCode(name='err')}


def create_dmda(target, objs):

    args = [objs['comm']]

    args += ['DM_BOUNDARY_GHOSTED' for _ in range(len(target.space_dimensions))]

    # Stencil type
    if len(target.space_dimensions) > 1:
        args += ['DMDA_STENCIL_BOX']

    # Global dimensions
    args += list(target.shape_global)[::-1]

    # No.of processors in each dimension
    if len(target.space_dimensions) > 1:
        args += list(target.grid.distributor.topology)[::-1]

    args += [1, target.space_order]

    args += [Null for _ in range(len(target.space_dimensions))]

    args += [Byref(objs['da'])]

    dmda = Call('PetscCall', [Call(f'DMDACreate{len(target.space_dimensions)}d',
                                   arguments=args)])

    return dmda


def build_solver_objs(target):
    
    return {'Jac': Mat(name='J_'+str(target.name)),
            'x_global': Vec(name='x_global_'+str(target.name)),
            'x_local': Vec(name='x_local_'+str(target.name), liveness='eager'),
            'b_global': Vec(name='b_global_'+str(target.name)),
            'b_local': Vec(name='b_local_'+str(target.name), liveness='eager'),
            'ksp': KSP(name='ksp_'+str(target.name)),
            'pc': PC(name='pc_'+str(target.name)),
            'snes': SNES(name='snes_'+str(target.name)),
            'x': Vec(name='x_'+str(target.name)),
            'y': Vec(name='y_'+str(target.name))}


def generate_solver_calls(solver_objs, objs, matvec):

    snes_create = Call('PetscCall', [Call('SNESCreate', arguments=[
        objs['comm'], Byref(solver_objs['snes'])])])
    
    snes_set_dm = Call('PetscCall', [Call('SNESSetDM', arguments=[
        solver_objs['snes'], objs['da']])])
    
    create_matrix = Call('PetscCall', [Call('DMCreateMatrix', arguments=[
        objs['da'], Byref(solver_objs['Jac'])])])
    
    # NOTE: Assumming all solves are linear for now.
    snes_set_type = Call('PetscCall', [Call('SNESSetType', arguments=[
        solver_objs['snes'], 'SNESKSPONLY'])])
    
    global_x = Call('PetscCall', [Call('DMCreateGlobalVector', arguments=[
        objs['da'], Byref(solver_objs['x_global'])])])
    
    local_x = Call('PetscCall', [Call('DMCreateLocalVector', arguments=[
        objs['da'], Byref(solver_objs['x_local'])])])
    
    global_b = Call('PetscCall', [Call('DMCreateGlobalVector', arguments=[
        objs['da'], Byref(solver_objs['b_global'])])])
    
    local_b = Call('PetscCall', [Call('DMCreateLocalVector', arguments=[
        objs['da'], Byref(solver_objs['b_local'])])])
    
    snes_get_ksp = Call('PetscCall', [Call('SNESGetKSP', arguments=[
        solver_objs['snes'], Byref(solver_objs['ksp'])])])
    
    return tuple([snes_create, snes_set_dm, create_matrix, snes_set_type,
                  global_x, local_x, global_b, local_b, snes_get_ksp])
    

def create_matvec_callback(target, matvec_callback_body_iters,
                           solver_objs, objs):

    matvec_callback = Callable('MyMatShellMult_'+str(target.name),
                            matvec_callback_body_iters,
                            retval=objs['err'],
                            parameters=(solver_objs['Jac'],
                                        solver_objs['x'],
                                        solver_objs['y']))
            
    matvec_operation = Call('PetscCall', [
        Call('MatShellSetOperation', arguments=[
            solver_objs['Jac'],
            'MATOP_MULT',
            Callback(matvec_callback.name, 'void', 'void')])])
    
    return matvec_callback, matvec_operation


Null = Macro('NULL')
