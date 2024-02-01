from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (List, Callable, Call, Transformer,
                           Callback, Definition, Uxreplace, FindSymbols,
                           Iteration, MapNodes, ActionExpr, RHSExpr,
                           SolutionExpr, FindNodes, Expression, PETScDumExpr,
                           retrieve_iteration_tree, filter_iterations, PreExpr)
from devito.types.petsc import (Mat, Vec, DM, PetscErrorCode, PETScStruct,
                                PETScArray, PetscMPIInt, KSP, PC)
from devito.symbolics import FieldFromPointer, Byref
import cgen as c


__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):
    # NOTE: Currently only considering the case when opt='noop'. This
    # does not deal with the Temp Expressions generated when opt is not set
    # to 'noop' etc.

    # Immediately drop all PETScDummyExprs.
    iet = drop_dummies(iet)

    # Generate the one off PETSc calls that do not need to be duplicated in the case of
    # multiple PETScSolves e.g PetscInitialize.
    init_setup = petsc_setup()

    # Build PETSc struct. Probably just want 1 struct for all PETScSolves. This may
    # be adjusted.
    struct = build_struct(iet)

    # Figure out how many PETScSolves were passed to the Operator.
    iter_sol_mapper = MapNodes(Iteration, SolutionExpr, 'groupby').visit(iet)

    setup = []
    efuncs = []

    # Generate the code required for each PETScSolve.
    for iter, (sol,) in iter_sol_mapper.items():

        # For each PETSc solve, build the corresponding PETSc objects.
        petsc_objs = build_petsc_objects(sol.target)

        solver_setup = build_solver_setup(petsc_objs, sol, struct)
        setup.extend(solver_setup)

        # This is the loop to pass the PETSc solution Vec x back to Devito.
        # TODO: Perhaps there is a cleaner way? But I'm not sure what yet.
        sol_iter = filter_iterations(iter, key=lambda i: i.dim.is_Space)

        mapper_main = {}
        for tree in retrieve_iteration_tree(iet):
            # from IPython import embed; embed()
            root = filter_iterations(tree, key=lambda i: i.dim.is_Space)

            # For each PETScsolve, there will be 1 ActionExpr, PreExpr and RHSExpr.
            action_expr = FindNodes(ActionExpr).visit(root)
            preconditioner_expr = FindNodes(PreExpr).visit(root)
            rhs_expr = FindNodes(RHSExpr).visit(root)

            if action_expr and action_expr[0].target.function == sol.target.function:

                # Build the body of the matvec callback for this PETScSolve.
                matvec_body = build_matvec_body(root[0], petsc_objs,
                                                struct, action_expr[0])
                mapper = {sol.target.function.indexed: petsc_objs['xvec_tmp'].indexed}
                matvec_body = Uxreplace(mapper).visit(matvec_body)

                # Create Callable for the matvec callback.
                matvec_callback = Callable('MyMatShellMult_'+str(sol.target.name),
                                           matvec_body,
                                           retval=petsc_objs['err'],
                                           parameters=(petsc_objs['A_matfree'],
                                                       petsc_objs['xvec'],
                                                       petsc_objs['yvec']))

                matvec_operation = Call('PetscCall', [
                    Call('MatShellSetOperation', arguments=[
                        petsc_objs['A_matfree'],
                        'MATOP_MULT',
                        Callback(matvec_callback.name, 'void', 'void')])])

                setup.append(matvec_operation)
                mapper_main.update({root[0]: None})
                efuncs.append(matvec_callback)

            elif preconditioner_expr and \
                    preconditioner_expr[0].target.function == sol.target.function:

                # Build the body of the preconditioner callback for this PETScSolve.
                pre_body = build_pre_body(root[0], petsc_objs,
                                          struct, preconditioner_expr[0])

                # Create Callable for the preconditioner callback.
                pre_callback = Callable('preconditioner_callback_'+str(sol.target.name),
                                        pre_body,
                                        retval=petsc_objs['err'],
                                        parameters=(petsc_objs['A_matfree'],
                                                    petsc_objs['yvec']))

                preconditioner_operation = Call('PetscCall', [
                    Call('MatShellSetOperation', arguments=[
                        petsc_objs['A_matfree'],
                        'MATOP_GET_DIAGONAL',
                        Callback(pre_callback.name, 'void', 'void')])])

                setup.append(preconditioner_operation)
                mapper_main.update({root[0]: None})
                efuncs.append(pre_callback)

            elif rhs_expr and rhs_expr[0].target.function == sol.target.function:

                solver_body = execute_solve(root[0], petsc_objs,
                                            rhs_expr[0], sol_iter[0], sol)
                mapper_main.update({root[0]: solver_body})

        mapper_main.update({sol_iter[0]: None})
        iet = Transformer(mapper_main).visit(iet)

    includes = []
    if iter_sol_mapper:
        body = iet.body._rebuild(body=(tuple(init_setup) + tuple(setup) + iet.body.body))
        iet = iet._rebuild(body=body)

        # TODO: Obviously it won't be like this.
        kwargs['compiler'].add_include_dirs('/home/zl5621/petsc/include')
        path = '/home/zl5621/petsc/arch-linux-c-debug/include'
        kwargs['compiler'].add_include_dirs(path)
        kwargs['compiler'].add_libraries('petsc')
        libdir = '/home/zl5621/petsc/arch-linux-c-debug/lib'
        kwargs['compiler'].add_library_dirs(libdir)
        kwargs['compiler'].add_ldflags('-Wl,-rpath,%s' % libdir)

        includes.extend(['petscksp.h', 'petscdmda.h'])

    return iet, {'efuncs': efuncs,
                 'includes': includes}


def drop_dummies(iet):

    # If a PETScDummyExpr is alongside non PETScDummyExprs
    # (e.g standard expressions) then just drop the PETScDummyExpr,
    # otherwise drop the entire iteration loop containing the PETScDummyExpr.

    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]

        dummy = FindNodes(PETScDumExpr).visit(root)
        if dummy:
            all_exprs = FindNodes(Expression).visit(root)
            if any(not isinstance(i, PETScDumExpr) for i in all_exprs):
                mapper.update({dummy[0]: None})
            else:
                mapper.update({root: None})

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


def petsc_setup():

    header = c.Line('PetscFunctionBeginUser;')

    size = PetscMPIInt(name='size')

    # TODO: This will obv change when it is working with MPI.
    initialize = Call('PetscCall', [Call('PetscInitialize',
                                         arguments=['NULL', 'NULL',
                                                    'NULL', 'NULL'])])

    call_mpi = Call('PetscCallMPI', [Call('MPI_Comm_size',
                                          arguments=['PETSC_COMM_WORLD',
                                                     Byref(size)])])

    return [header, initialize, call_mpi, c.Line()]


def build_petsc_objects(target):
    # TODO: Eventually, the objects built will be based
    # on the number of different PETSc equations present etc.

    return {'A_matfree': Mat(name='A_matfree_'+str(target.name)),
            'xvec': Vec(name='xvec_'+str(target.name)),
            'local_xvec': Vec(name='local_xvec_'+str(target.name), liveness='eager'),
            'yvec': Vec(name='yvec_'+str(target.name)),
            'da': DM(name='da_'+str(target.name), liveness='eager'),
            'x': Vec(name='x_'+str(target.name)),
            'b': Vec(name='b_'+str(target.name)),
            'ksp': KSP(name='ksp_'+str(target.name)),
            'pc': PC(name='pc_'+str(target.name)),
            'err': PetscErrorCode(name='err_'+str(target.name)),
            'reason': PetscErrorCode(name='reason_'+str(target.name)),
            'xvec_tmp': (PETScArray(name='xvec_tmp_'+str(target.name), dtype=target.dtype,
                                    dimensions=target.dimensions,
                                    shape=target.shape,
                                    liveness='eager'))}


def build_struct(iet):

    # Place all symbols required by all PETSc solves into the same struct.
    usr_ctx = []

    tmp1 = FindSymbols('basics').visit(iet)
    tmp2 = FindSymbols('dimensions|indexedbases').visit(iet)
    usr_ctx.extend(symb for symb in tmp1 if symb not in tmp2)

    return PETScStruct('ctx', usr_ctx)


def build_matvec_body(action, objs, struct, action_expr):

    get_context = Call('PetscCall', [Call('MatShellGetContext',
                                          arguments=[objs['A_matfree'],
                                                     Byref(struct)])])

    mat_get_dm = Call('PetscCall', [Call('MatGetDM',
                                         arguments=[objs['A_matfree'],
                                                    Byref(objs['da'])])])

    dm_get_local_xvec = Call('PetscCall', [Call('DMGetLocalVector',
                                                arguments=[objs['da'],
                                                           Byref(objs['local_xvec'])])])

    dm_global_local_begin = Call('PetscCall', [Call('DMGlobalToLocalBegin',
                                                    arguments=[objs['da'],
                                                               objs['xvec'],
                                                               'INSERT_VALUES',
                                                               objs['local_xvec']])])

    dm_global_local_end = Call('PetscCall', [Call('DMGlobalToLocalEnd',
                                                  arguments=[objs['da'],
                                                             objs['xvec'],
                                                             'INSERT_VALUES',
                                                             objs['local_xvec']])])

    dm_vec_get_array_read = Call('PetscCall',
                                 [Call('DMDAVecGetArrayRead',
                                       arguments=[objs['da'],
                                                  objs['local_xvec'],
                                                  Byref(objs['xvec_tmp']._C_symbol)])])

    dm_vec_get_array = Call('PetscCall',
                            [Call('DMDAVecGetArray',
                                  arguments=[objs['da'],
                                             objs['yvec'],
                                             Byref(action_expr.write._C_symbol)])])

    dm_vec_restore_array_read = Call(
        'PetscCall',
        [Call('DMDAVecRestoreArrayRead', arguments=[objs['da'],
                                                    objs['local_xvec'],
                                                    Byref(objs['xvec_tmp']._C_symbol)])])

    dm_vec_restore_array = Call(
        'PetscCall',
        [Call('DMDAVecRestoreArray', arguments=[objs['da'],
                                                objs['yvec'],
                                                Byref(action_expr.write._C_symbol)])])

    dm_restore_local_vec = Call(
        'PetscCall',
        [Call('DMRestoreLocalVector', arguments=[objs['da'],
                                                 Byref(objs['local_xvec'])])])

    func_return = Call('PetscFunctionReturn', arguments=[0])

    body = List(header=c.Line('PetscFunctionBegin;'),
                body=[Definition(struct),
                      get_context,
                      mat_get_dm,
                      dm_get_local_xvec,
                      dm_global_local_begin,
                      dm_global_local_end,
                      dm_vec_get_array_read,
                      dm_vec_get_array,
                      action,
                      # TODO: Track BCs through PETScSolve This line will come
                      # from the BCs.
                      c.Line('y_matvec_pn1[0][0]= xvec_tmp_pn1[0][0];'),
                      dm_vec_restore_array_read,
                      dm_vec_restore_array,
                      dm_restore_local_vec,
                      func_return])

    # Replace all symbols in the body that appear in the struct
    # with a pointer to the struct.
    for i in struct.usr_ctx:
        body = Uxreplace({i: FieldFromPointer(i, struct)}).visit(body)

    return body


ksp_mapper = {
    'gmres': 'KSPGMRES',
}

pc_mapper = {
    'jacobi': 'PCJACOBI',
}


# def build_solve_setup(matvec_body, pre_body, objs, struct, target):
def build_solver_setup(objs, sol, struct):

    # TODO: Create DM based on the dimensions of the target field i.e
    # this determines DMDACreate2d, 3d etc
    dm_create = Call('PetscCall', [Call('DMDACreate2d',
                                        arguments=['PETSC_COMM_SELF',
                                                   'DM_BOUNDARY_MIRROR',
                                                   'DM_BOUNDARY_MIRROR',
                                                   'DMDA_STENCIL_STAR',
                                                   sol.target.shape[0],
                                                   sol.target.shape[1],
                                                   'PETSC_DECIDE',
                                                   'PETSC_DECIDE',
                                                   1, 1,
                                                   'NULL', 'NULL',
                                                   Byref(objs['da'])])])

    dm_set_from_options = Call('PetscCall', [Call('DMSetFromOptions',
                                                  arguments=[objs['da']])])

    dm_setup = Call('PetscCall', [Call('DMSetUp', arguments=[objs['da']])])

    dm_set_mattype = Call('PetscCall', [Call('DMSetMatType',
                                             arguments=[objs['da'],
                                                        'MATSHELL'])])

    dm_create_mat = Call('PetscCall', [Call('DMCreateMatrix',
                                            arguments=[objs['da'],
                                                       Byref(objs['A_matfree'])])])

    set_context = Call('PetscCall', [Call('MatShellSetContext',
                                          arguments=[objs['A_matfree'],
                                                     struct])])

    dm_create_global_vec_x = Call('PetscCall', [Call('DMCreateGlobalVector',
                                                     arguments=[objs['da'],
                                                                Byref(objs['x'])])])
    dm_create_global_vec_b = Call('PetscCall', [Call('DMCreateGlobalVector',
                                                     arguments=[objs['da'],
                                                                Byref(objs['b'])])])

    ksp_create = Call('PetscCall', [Call('KSPCreate',
                                         arguments=['PETSC_COMM_SELF',
                                                    Byref(objs['ksp'])])])

    ksp_set_operators = Call('PetscCall', [Call('KSPSetOperators',
                                                arguments=[objs['ksp'],
                                                           objs['A_matfree'],
                                                           objs['A_matfree']])])

    rtol = sol.solver_parameters.get('ksp_rtol', 'PETSC_DEFAULT')
    abstol = sol.solver_parameters.get('ksp_atol', 'PETSC_DEFAULT')
    divtol = sol.solver_parameters.get('ksp_divtol', 'PETSC_DEFAULT')
    max_its = sol.solver_parameters.get('ksp_max_it', 'PETSC_DEFAULT')

    ksp_set_tol = Call('PetscCall', [Call('KSPSetTolerances',
                                          arguments=[objs['ksp'],
                                                     rtol,
                                                     abstol,
                                                     divtol,
                                                     max_its])])

    # Set default KSP type to GMRES
    ksp_type = ksp_mapper[sol.solver_parameters.get('ksp_type', 'gmres')]
    ksp_set_type = Call('PetscCall', [Call('KSPSetType',
                                           arguments=[objs['ksp'],
                                                      ksp_type])])

    ksp_get_pc = Call('PetscCall', [Call('KSPGetPC',
                                         arguments=[objs['ksp'],
                                                    Byref(objs['pc'])])])

    # NOTE: Obvs temporary, but I'm setting the default preconditioner to
    # JACOBI diagonal for now.
    pc_type = pc_mapper[sol.solver_parameters.get('pc_type', 'jacobi')]
    pc_set_type = Call('PetscCall', [Call('PCSetType',
                                          arguments=[objs['pc'],
                                                     pc_type])])
    if pc_type == 'PCJACOBI':
        pc_jacobi_set_type = Call('PetscCall', [Call('PCJacobiSetType',
                                                     arguments=[objs['pc'],
                                                                'PC_JACOBI_DIAGONAL'])])

    ksp_set_from_opts = Call('PetscCall', [Call('KSPSetFromOptions',
                                                arguments=[objs['ksp']])])

    body = [dm_create, dm_set_from_options, dm_setup, dm_set_mattype, dm_create_mat,
            set_context, dm_create_global_vec_x, dm_create_global_vec_b, ksp_create,
            ksp_set_operators, ksp_set_tol, ksp_set_type, ksp_get_pc, pc_set_type,
            pc_jacobi_set_type, ksp_set_from_opts]

    return body


def execute_solve(rhs_iter, objs, rhs_expr, sol_iter, sol):

    dm_vec_get_array = Call('PetscCall', [Call('DMDAVecGetArray',
                                               arguments=[objs['da'],
                                                          objs['b'],
                                                          Byref(rhs_expr.write)])])

    dm_vec_restore_array_b = Call('PetscCall', [Call('DMDAVecRestoreArray',
                                                     arguments=[objs['da'],
                                                                objs['b'],
                                                                Byref(rhs_expr.write)])])

    ksp_solve = Call('PetscCall', [Call('KSPSolve',
                                        arguments=[objs['ksp'],
                                                   objs['b'],
                                                   objs['x']])])

    dm_vec_get_array_x = Call('PetscCall', [Call('DMDAVecGetArray',
                                                 arguments=[objs['da'],
                                                            objs['x'],
                                                            Byref(sol.reads[0])])])

    dm_vec_restore_array_x = Call('PetscCall', [Call('DMDAVecRestoreArray',
                                                     arguments=[objs['da'],
                                                                objs['x'],
                                                                Byref(sol.reads[0])])])

    # TODO: DESTROY OBJECTS

    body = List(body=[dm_vec_get_array,
                      rhs_iter,
                      dm_vec_restore_array_b,
                      ksp_solve,
                      dm_vec_get_array_x,
                      sol_iter,
                      dm_vec_restore_array_x])

    return body


def build_pre_body(pre_iteration, objs, struct, expr_target):

    get_context = Call('PetscCall', [Call('MatShellGetContext',
                                          arguments=[objs['A_matfree'],
                                                     Byref(struct)])])

    mat_get_dm = Call('PetscCall', [Call('MatGetDM',
                                         arguments=[objs['A_matfree'],
                                                    Byref(objs['da'])])])

    dm_vec_get_array = Call('PetscCall',
                            [Call('DMDAVecGetArray',
                                  arguments=[objs['da'],
                                             objs['yvec'],
                                             Byref(expr_target.write._C_symbol)])])

    dm_vec_restore_array = Call(
        'PetscCall',
        [Call('DMDAVecRestoreArray', arguments=[objs['da'],
                                                objs['yvec'],
                                                Byref(expr_target.write._C_symbol)])])

    func_return = Call('PetscFunctionReturn', arguments=[0])

    body = List(header=c.Line('PetscFunctionBegin;'),
                body=[Definition(struct),
                      get_context,
                      mat_get_dm,
                      dm_vec_get_array,
                      pre_iteration,
                      # TODO: Track BCs through PETScSolve This line
                      # will come from the BCs.
                      c.Line('y_pre_pn1[0][0]=1.;'),
                      dm_vec_restore_array,
                      func_return])

    # Replace all symbols in the body that appear in the struct
    # with a pointer to the struct.
    for i in struct.usr_ctx:
        body = Uxreplace({i: FieldFromPointer(i, struct)}).visit(body)
    return body
