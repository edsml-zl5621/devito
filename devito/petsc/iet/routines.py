from collections import OrderedDict

import cgen as c
import numpy as np

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, BlankLine, Callable, FindNodes,
                           retrieve_iteration_tree, filter_iterations, Iteration)
from devito.symbolics import (Byref, FieldFromPointer, Macro, cast_mapper,
                              FieldFromComposite, IntDiv, Mod)
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import Temp, Symbol, CustomDimension, Dimension
from devito.tools import filter_ordered

from devito.petsc.types import PETScArray, PETScStruct
from devito.petsc.iet.nodes import (PETScCallable, FormFunctionCallback,
                                    MatShellSetOp, InjectSolveDummy)
from devito.petsc.iet.utils import petsc_call, petsc_struct
from devito.petsc.utils import solver_mapper
from devito.petsc.types import (DM, CallbackDM, Mat, LocalVec, GlobalVec, KSP, PC,
                                SNES, DummyArg, PetscInt, StartPtr, SingleIS, IS, SubDM, SubMats)


class CBBuilder:
    """
    Build IET routines to generate PETSc callback functions.
    """
    def __init__(self, injectsolve, objs, solver_objs,
                 rcompile=None, sregistry=None, timedep=None, **kwargs):

        self.rcompile = rcompile
        self.sregistry = sregistry
        self.timedep = timedep
        self.objs = objs
        self.solver_objs = solver_objs
        self.injectsolve = injectsolve

        self._efuncs = OrderedDict()
        self._struct_params = {}

        self._matvec_callback = None
        self._matvecs = None
        self._formfunc_callback = None
        self._formrhs_callback = None
        self._struct_callback = None

        self._make_core()
        self._main_struct()
        self._make_struct_callback()
        self._efuncs = self._uxreplace_efuncs()

    @property
    def efuncs(self):
        return self._efuncs

    @property
    def struct_params(self):
        return self._struct_params

    @property
    def filtered_struct_params(self):
        """
        Return ordered, filtered struct parameters, grouped by submatrix.
        """
        return {key: filter_ordered(params) for key, params in self.struct_params.items()}

    def add_struct_params(self, submatrix_name, params):
        """
        Add struct parameters for a specific submatrix.
        """
        if submatrix_name not in self._struct_params:
            self._struct_params[submatrix_name] = []
        self._struct_params[submatrix_name].extend(params)

    @property
    def matvec_callback(self):
        """
        This is the matvec callback associated with the whole Jacobian i.e
        is set in the main kernel via
        `PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))MyMatShellMult));`
        """
        return self._matvecs

    @property
    def matvecs(self):
        return self._matvecs

    @property
    def formfunc_callback(self):
        return self._formfunc_callback

    @property
    def formrhs_callback(self):
        return self._formrhs_callback

    @property
    def struct_callback(self):
        return self._struct_callback

    def _make_core(self):
        fielddata = self.injectsolve.expr.rhs.fielddata
        self._make_matvec(fielddata)
        self._make_formfunc(fielddata)
        self._make_formrhs(fielddata)

    def _make_matvec(self, fielddata):
        # Compile matvec `eqns` into an IET via recursive compilation
        matvecs = fielddata.matvecs
        irs_matvec, _ = self.rcompile(
            matvecs, options={'mpi': False}, sregistry=self.sregistry
        )
        body_matvec = self._create_matvec_body(List(body=irs_matvec.uiet.body),
                                               fielddata)
        sobjs = self.solver_objs
        matvec_callback = PETScCallable(
            self.sregistry.make_name(prefix='MyMatShellMult_'), body_matvec,
            retval=self.objs['err'],
            parameters=(
                sobjs['Jac'], sobjs['X_global'], sobjs['Y_global']
            )
        )
        self._matvecs = matvec_callback
        self._efuncs[matvec_callback.name] = matvec_callback

    def _create_matvec_body(self, body, data):
        linsolve_expr = self.injectsolve.expr.rhs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)

        # TODO: maybe this shouldn't be attached to the fielddata -> think about this
        # currently it's attched to both i think
        y_matvec = data.arrays['y_matvec']
        x_matvec = data.arrays['x_matvec']

        mat_get_dm = petsc_call('MatGetDM', [sobjs['Jac'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['X_local'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, sobjs['X_global'],
                                     'INSERT_VALUES', sobjs['X_local']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, sobjs['X_global'], 'INSERT_VALUES', sobjs['X_local']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['Y_local'])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [sobjs['Y_local'], Byref(y_matvec._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [sobjs['X_local'], Byref(x_matvec._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [sobjs['Y_local'], Byref(y_matvec._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [sobjs['X_local'], Byref(x_matvec._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, sobjs['Y_local'], 'INSERT_VALUES', sobjs['Y_global']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, sobjs['Y_local'], 'INSERT_VALUES', sobjs['Y_global']
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(sobjs['X_local'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(sobjs['Y_local'])]
        )

        # TODO: Some of the calls are placed in the `stacks` argument of the
        # `CallableBody` to ensure that they precede the `cast` statements. The
        # 'casts' depend on the calls, so this order is necessary. By doing this,
        # you avoid having to manually construct the `casts` and can allow
        # Devito to handle their construction. This is a temporary solution and
        # should be revisited

        body = body._rebuild(
            body=body.body +
            (vec_restore_array_y,
             vec_restore_array_x,
             dm_local_to_global_begin,
             dm_local_to_global_end,
             dm_restore_local_xvec,
             dm_restore_local_yvec)
        )

        stacks = (
            mat_get_dm,
            dm_get_app_context,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            vec_get_array_y,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, dummyctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        matvec_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, dummyctx) for i in fields}
        matvec_body = Uxreplace(subs).visit(matvec_body)

        self.add_struct_params("J00", fields)
        # from IPython import embed; embed()
        return matvec_body

    # TODO: think can remove "objs" as an argument to alot of these functions
    def _make_formfunc(self, fielddata):
        # Compile formfunc `eqns` into an IET via recursive compilation
        formfuncs = fielddata.formfuncs
        irs_formfunc, _ = self.rcompile(
            formfuncs, options={'mpi': False}, sregistry=self.sregistry
        )
        body_formfunc = self._create_formfunc_body(
            List(body=irs_formfunc.uiet.body), fielddata
        )
        sobjs = self.solver_objs
        formfunc_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormFunction_'), body_formfunc,
            retval=self.objs['err'],
            parameters=(sobjs['snes'], sobjs['X_global'],
                        sobjs['F_global'], dummyptr)
        )
        self._formfunc_callback = formfunc_callback
        self._efuncs[formfunc_callback.name] = formfunc_callback

    def _create_formfunc_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)

        f_formfunc = fielddata.arrays['f_formfunc']
        x_formfunc = fielddata.arrays['x_formfunc']

        snes_get_dm = petsc_call('SNESGetDM', [sobjs['snes'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['X_local'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, sobjs['X_global'],
                                     'INSERT_VALUES', sobjs['X_local']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, sobjs['X_global'], 'INSERT_VALUES', sobjs['X_local']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['F_local'])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [sobjs['F_local'], Byref(f_formfunc._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [sobjs['X_local'], Byref(x_formfunc._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [sobjs['F_local'], Byref(f_formfunc._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [sobjs['X_local'], Byref(x_formfunc._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, sobjs['F_local'], 'INSERT_VALUES', sobjs['F_global']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, sobjs['F_local'], 'INSERT_VALUES', sobjs['F_global']
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(sobjs['X_local'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(sobjs['F_local'])]
        )

        body = body._rebuild(
            body=body.body +
            (vec_restore_array_y,
             vec_restore_array_x,
             dm_local_to_global_begin,
             dm_local_to_global_end,
             dm_restore_local_xvec,
             dm_restore_local_yvec)
        )

        stacks = (
            snes_get_dm,
            dm_get_app_context,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            vec_get_array_y,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, dummyctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, dummyctx) for i in fields}
        formfunc_body = Uxreplace(subs).visit(formfunc_body)

        # self._struct_params.extend(fields)
        self.add_struct_params("J00", fields)

        return formfunc_body

    def _make_formrhs(self, fielddata):
        # Compile formrhs `eqns` into an IET via recursive compilation
        formrhs = fielddata.formrhs
        irs_formrhs, _ = self.rcompile(
            formrhs, options={'mpi': False}, sregistry=self.sregistry
        )
        body_formrhs = self._create_formrhs_body(
            List(body=irs_formrhs.uiet.body), fielddata
        )
        formrhs_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormRHS_'), body_formrhs, retval=self.objs['err'],
            parameters=(
                self.solver_objs['snes'], self.solver_objs['b_local']
            )
        )
        self._formrhs_callback = formrhs_callback
        self._efuncs[formrhs_callback.name] = formrhs_callback

    def _create_formrhs_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']

        snes_get_dm = petsc_call('SNESGetDM', [sobjs['snes'], Byref(dmda)])

        b_arr = fielddata.arrays['b_tmp']

        vec_get_array = petsc_call(
            'VecGetArray', [sobjs['b_local'], Byref(b_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        vec_restore_array = petsc_call(
            'VecRestoreArray', [sobjs['b_local'], Byref(b_arr._C_symbol)]
        )

        body = body._rebuild(body=body.body + (vec_restore_array,))

        stacks = (
            snes_get_dm,
            dm_get_app_context,
            vec_get_array,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, dummyctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        formrhs_body = CallableBody(
            List(body=[body]),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, dummyctx) for
                i in fields if not isinstance(i.function, AbstractFunction)}
        formrhs_body = Uxreplace(subs).visit(formrhs_body)

        self.add_struct_params("J00", fields)

        return formrhs_body

    def local_struct(self):
        """
        This is the struct used within callback functions,
        usually accessed via DMGetApplicationContext.
        """
        # from IPython import embed; embed()
        # TODO: can probs drop liveness now?
        localctx = petsc_struct(
            dummyctx.name,
            self.filtered_struct_params['J00'],
            self.solver_objs['Jac'].name+'_ctx',
            liveness='eager', modifier=' *'
        )
        # localctx = PETScStruct(localctx.name, localctx.dtype, localctx.value)
        return localctx

    def _main_struct(self):
        """
        This is the struct initialised inside the main kernel and
        attached to the DM via DMSetApplicationContext.
        """
        self.solver_objs['mainctx'] = petsc_struct(
            self.sregistry.make_name(prefix='ctx'),
            self.filtered_struct_params['J00'],
            self.solver_objs['Jac'].name+'_ctx'
        )

    def _make_struct_callback(self):
        mainctx = self.solver_objs['mainctx']

        body = [
            DummyExpr(FieldFromPointer(i._C_symbol, mainctx), i._C_symbol)
            for i in mainctx.callback_fields
        ]
        struct_callback_body = CallableBody(
            List(body=body), init=(petsc_func_begin_user,),
            retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
        )
        struct_callback = Callable(
            self.sregistry.make_name(prefix='PopulateMatContext_'),
            struct_callback_body, self.objs['err'],
            parameters=[mainctx]
        )
        self._efuncs[struct_callback.name] = struct_callback
        self._struct_callback = struct_callback

    def _dummy_fields(self, iet):
        # Place all context data required by the shell routines into a struct
        fields = [f.function for f in FindSymbols('basics').visit(iet)]
        fields = [f for f in fields if not isinstance(f.function, (PETScArray, Temp))]
        fields = [
            f for f in fields if not (f.is_Dimension and not (f.is_Time or f.is_Modulo))
        ]
        return fields

    def _uxreplace_efuncs(self):
        # from IPython import embed; embed()
        lstruct = self.local_struct()
        mapper = {}
        visitor = Uxreplace({dummyctx: lstruct})
        for k, v in self._efuncs.items():
            mapper.update({k: visitor.visit(v)})
        return mapper


class CCBBuilder(CBBuilder):

    def __init__(self, injectsolve, objs, solver_objs, **kwargs):
        self._submatrices_callback = None
        super().__init__(injectsolve, objs, solver_objs, **kwargs)
        self._coupled_struct_callback = None
        self._populate_coupled_ctx()

    @property
    def submatrices_callback(self):
        return self._submatrices_callback

    @property
    def coupled_struct_callback(self):
        return self._coupled_struct_callback

    def _make_core(self):
        # let's just start by generating the diagonal sub matrices, then will extend to off diags
        injectsolve = self.injectsolve
        targets = injectsolve.expr.rhs.fielddata.targets
        all_fielddata = injectsolve.expr.rhs.fielddata 

        # for t in targets:
        #     data = all_fielddata.get_field_data(t)
        #     self._make_matvec(objs, solver_objs)

        data = all_fielddata.get_field_data(targets[0])
        # from IPython import embed; embed()
        self._make_matvec(data)
        self._make_formfunc(data)
        self._make_formrhs(data)

        self._create_submatrices()

    def _create_submatrices(self):
        body = self._create_submat_callback_body()
        sobjs = self.solver_objs

        submatrices_callback = PETScCallable(
            self.sregistry.make_name(prefix='MatCreateSubMatrices_'), body,
            retval=self.objs['err'],
            parameters=(
                sobjs['Jac'], sobjs['n_submats'], sobjs['submats']
            )
        )
        self._submatrices_callback = submatrices_callback
        self._efuncs[submatrices_callback.name] = submatrices_callback

    # TODO: obvs improve these names
    def _create_submat_callback_body(self):
        sobjs = self.solver_objs

        mat_get_dm = petsc_call('MatGetDM', [sobjs['Jac'], Byref(sobjs['callbackdm'])])

        shell_get_ctx = petsc_call('MatShellGetContext', [sobjs['Jac'], Byref(sobjs['ljacctx']._C_symbol)])

        dm_get_info = petsc_call('DMGetInfo', [sobjs['callbackdm'], Null, Byref(sobjs['M']), Byref(sobjs['N']), Null, Null, Null, Null, Byref(sobjs['dof']), Null, Null, Null, Null, Null])
        
        subblock_rows = DummyExpr(sobjs['subblockrows'], Mul(sobjs['M'], sobjs['N']))
        subblock_cols = DummyExpr(sobjs['subblockcols'], Mul(sobjs['M'], sobjs['N']))

        mat_create = petsc_call('MatCreate', [self.objs['comm'], Byref(sobjs['block'])])
        mat_set_sizes = petsc_call('MatSetSizes', [sobjs['block'], 'PETSC_DECIDE', 'PETSC_DECIDE', sobjs['subblockrows'], sobjs['subblockcols']])
        mat_set_type = petsc_call('MatSetType', [sobjs['block'], 'MATSHELL'])

        malloc = petsc_call('PetscMalloc', [1, Byref(sobjs['submat_ctx'])])
        i = Dimension(name='i')

        row_idx = DummyExpr(sobjs['row_idx'], IntDiv(i, sobjs['dof']))
        col_idx = DummyExpr(sobjs['col_idx'], Mod(i, sobjs['dof']))

        deref_subdm = Dereference(sobjs['subdms'], sobjs['ljacctx'])

        set_dm = DummyExpr(FieldFromPointer(sobjs['subdm'], sobjs['submat_ctx']), sobjs['subdms'].indexed[sobjs['row_idx']])
        # fix:todo: the SUBMAT_CTX doesn't appear in the ccode because it's not an argument to any function -> fix this in the cgen structure code
        set_rows = DummyExpr(FieldFromPointer(sobjs['rows'], sobjs['submat_ctx']), Byref(sobjs['all_IS_rows'].indexed[sobjs['row_idx']]))

        iteration = Iteration(List(body=[mat_create, mat_set_sizes, mat_set_type, malloc, row_idx, col_idx, set_dm, set_rows]), i, sobjs['n_submats']-1)


        body = [mat_get_dm, dm_get_info, subblock_rows, subblock_cols, BlankLine, iteration]
        body = CallableBody(
            List(body=tuple(body)),
            init=(petsc_func_begin_user,),
            stacks=(shell_get_ctx, deref_subdm),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))
        return body

    def _populate_coupled_ctx(self):
        coupled_ctx = self.solver_objs['jacctx']

        body = [
            DummyExpr(FieldFromPointer(i._C_symbol, coupled_ctx), i._C_symbol)
            for i in coupled_ctx.callback_fields
        ]
        body = CallableBody(
            List(body=body), init=(petsc_func_begin_user,),
            retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
        )
        cb = Callable(
            self.sregistry.make_name(prefix='PopulateCoupledContext_'),
            body, self.objs['err'],
            parameters=[coupled_ctx]+coupled_ctx.callback_fields
        )
        self._efuncs[cb.name] = cb
        self._coupled_struct_callback = cb
        # self._struct_callback = cb



class BaseObjectBuilder:
    """
    A base class for constructing objects needed for a PETSc solver.
    Designed to be extended by subclasses, which can override the `_extend_build`
    method to support specific use cases.
    """

    def __init__(self, injectsolve, sregistry=None, **kwargs):
        self.sregistry = sregistry
        self.fielddata = injectsolve.expr.rhs.fielddata
        self.solver_objs = self._build(injectsolve)

    def _build(self, injectsolve):
        """
        Constructs the core dictionary of solver objects and allows
        subclasses to extend or modify it via `_extend_build`.

        Returns:
            dict: A dictionary containing the following objects:
                - 'Jac' (Mat): A matrix representing the jacobian.
                - 'x_global' (GlobalVec): The global solution vector.
                - 'x_local' (LocalVec): The local solution vector.
                - 'b_global': (GlobalVec) Global RHS vector `b`, where `F(x) = b`.
                - 'b_local': (LocalVec) Local RHS vector `b`, where `F(x) = b`.
                - 'ksp': (KSP) Krylov solver object that manages the linear solver.
                - 'pc': (PC) Preconditioner object.
                - 'snes': (SNES) Nonlinear solver object.
                - 'F_global': (GlobalVec) Global residual vector `F`, where `F(x) = b`.
                - 'F_local': (LocalVec) Local residual vector `F`, where `F(x) = b`.
                - 'Y_global': (GlobalVector) The output vector populated by the
                   matrix-free `MyMatShellMult` callback function.
                - 'Y_local': (LocalVector) The output vector populated by the matrix-free
                   `MyMatShellMult` callback function.
                - 'X_global': (GlobalVec) Current guess for the solution,
                   required by the FormFunction callback.
                - 'X_local': (LocalVec) Current guess for the solution,
                   required by the FormFunction callback.
                - 'localsize' (PetscInt): The local length of the solution vector.
                - 'start_ptr' (StartPtr): A pointer to the beginning of the solution array
                   that will be updated at each time step.
                - 'dmda' (DM): The DMDA object associated with this solve, linked to
                   the SNES object via `SNESSetDM`.
                - 'callbackdm' (CallbackDM): The DM object accessed within callback
                   functions via `SNESGetDM`.
        """
        sreg = self.sregistry
        base_dict = {
            'Jac': Mat(sreg.make_name(prefix='J_')),
            'x_global': GlobalVec(sreg.make_name(prefix='x_global_')),
            'x_local': LocalVec(sreg.make_name(prefix='x_local_'), liveness='eager'),
            'b_global': GlobalVec(sreg.make_name(prefix='b_global_')),
            'b_local': LocalVec(sreg.make_name(prefix='b_local_')),
            'ksp': KSP(sreg.make_name(prefix='ksp_')),
            'pc': PC(sreg.make_name(prefix='pc_')),
            'snes': SNES(sreg.make_name(prefix='snes_')),
            'F_global': GlobalVec(sreg.make_name(prefix='F_global_')),
            'F_local': LocalVec(sreg.make_name(prefix='F_local_'), liveness='eager'),
            'Y_global': GlobalVec(sreg.make_name(prefix='Y_global_')),
            'Y_local': LocalVec(sreg.make_name(prefix='Y_local_'), liveness='eager'),
            'X_global': GlobalVec(sreg.make_name(prefix='X_global_')),
            'X_local': LocalVec(sreg.make_name(prefix='X_local_'), liveness='eager'),
            'localsize': PetscInt(sreg.make_name(prefix='localsize_')),
            'dmda': DM(sreg.make_name(prefix='da_'), liveness='eager',
                       stencil_width=self.fielddata.space_order),
            'callbackdm': CallbackDM(
                sreg.make_name(prefix='dm_'), liveness='eager',
                stencil_width=self.fielddata.space_order
            ),
        }
        base_dict = self._per_target(base_dict)
        return self._extend_build(base_dict, injectsolve)

    def _per_target(self, base_dict):
        sreg = self.sregistry
        targets = self.fielddata.targets
        for target in targets:
            base_dict[target.name+'_ptr'] = StartPtr(
                sreg.make_name(prefix='%s_ptr' % target.name), target.dtype
            )
        return base_dict

    def _extend_build(self, base_dict, injectsolve):
        """
        Subclasses can override this method to extend or modify the
        base dictionary of solver objects.
        """
        return base_dict


class CoupledObjectBuilder(BaseObjectBuilder):
    def _extend_build(self, base_dict, injectsolve):
        # TODO: add a no_of_targets attribute to the FieldData object
        no_targets = len(self.fielddata.targets)
        base_dict['fields'] = IS(
            name=self.sregistry.make_name(prefix='fields_'), nindices=no_targets
            )
        base_dict['subdms'] = SubDM(
            name=self.sregistry.make_name(prefix='subdms_'), nindices=no_targets
            )
        # CHANGE THIS TO PETSCINT
        base_dict['n_submats'] = Symbol(self.sregistry.make_name(prefix='n_submats_'), dtype=np.int32)
        base_dict['submats'] = SubMats(name=self.sregistry.make_name(prefix='submats_'),
                                       nindices=no_targets*no_targets)

        fields = [base_dict['n_submats'], base_dict['subdms'], base_dict['fields'], base_dict['snes']]
        # from IPython import embed; embed()
        # fields = [injectsolve.expr.rhs.fielddata.targets[0].grid.spacing_symbols[0]]
        # fields = [base_dict['snes']]

        # base_dict['jacctx'] = petsc_struct('whole_jac',
        #     fields,
        #     'JacobianContext',
        # )
        # base_dict['ljacctx'] = petsc_struct('whole_jac_local',
        #     fields,
        #     'JacobianContext',
        #     liveness='eager'
        # )
        #  TODO: use petsc_struct -> adapt it
        base_dict['jacctx'] = PETScStruct(
            name='whole_jac', pname='JacobianContext',
            fields=fields, liveness='lazy'
        )
        base_dict['ljacctx'] = PETScStruct(
            name='whole_jac_local', pname='JacobianContext',
            fields=fields, modifier=' *', liveness='eager'
        )

        # global submatrix sizes
        base_dict['M'] = PetscInt(self.sregistry.make_name(prefix='M_'))
        base_dict['N'] = PetscInt(self.sregistry.make_name(prefix='N_'))
        # from IPython import embed; embed()
        base_dict['dof'] = PetscInt(self.sregistry.make_name(prefix='dof_'))
        base_dict['block'] = Mat(self.sregistry.make_name(prefix='block_'))
        base_dict['subblockrows'] = PetscInt(self.sregistry.make_name(prefix='subblockrows_'))
        base_dict['subblockcols'] = PetscInt(self.sregistry.make_name(prefix='subblockcols_'))

        base_dict['all_IS_rows'] = IS(name=self.sregistry.make_name(prefix='allrows_'), nindices=no_targets)
        base_dict['all_IS_cols'] = IS(name=self.sregistry.make_name(prefix='allcols_'), nindices=no_targets)

        # the single IS rows owned by each submatrix
        base_dict['rows'] = SingleIS(name=self.sregistry.make_name(prefix='rows_'))
        base_dict['cols'] = SingleIS(name=self.sregistry.make_name(prefix='cols_'))


        # probably can just use the existing 'callbackdm'
        base_dict['subdm'] = DM(self.sregistry.make_name(prefix='subdm_'), liveness='eager')

        submatrix_ctx_fields = [base_dict['rows'], base_dict['cols'], base_dict['subdm']]
        base_dict['submat_ctx'] = PETScStruct(
            name='submat_ctx', pname='SubMatrixCtx',
            fields=submatrix_ctx_fields, modifier=' *', liveness='eager'
        )

        base_dict['row_idx'] = PetscInt(self.sregistry.make_name(prefix='row_idx_'))
        base_dict['col_idx'] = PetscInt(self.sregistry.make_name(prefix='col_idx_'))
        
        return base_dict


class BaseSetup:
    def __init__(self, solver_objs, objs, injectsolve, cbbuilder):
        self.injectsolve = injectsolve
        self.calls = self._setup(solver_objs, objs, injectsolve, cbbuilder)

    def _setup(self, solver_objs, objs, injectsolve, cbbuilder):
        dmda = solver_objs['dmda']

        solver_params = injectsolve.expr.rhs.solver_parameters

        snes_create = petsc_call('SNESCreate', [objs['comm'], Byref(solver_objs['snes'])])

        snes_set_dm = petsc_call('SNESSetDM', [solver_objs['snes'], dmda])

        create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(solver_objs['Jac'])])

        # NOTE: Assuming all solves are linear for now.
        snes_set_type = petsc_call('SNESSetType', [solver_objs['snes'], 'SNESKSPONLY'])

        snes_set_jac = petsc_call(
            'SNESSetJacobian', [solver_objs['snes'], solver_objs['Jac'],
                                solver_objs['Jac'], 'MatMFFDComputeJacobian', Null]
        )

        global_x = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(solver_objs['x_global'])])

        global_b = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(solver_objs['b_global'])])

        local_b = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(solver_objs['b_local'])])

        snes_get_ksp = petsc_call('SNESGetKSP',
                                  [solver_objs['snes'], Byref(solver_objs['ksp'])])

        ksp_set_tols = petsc_call(
            'KSPSetTolerances', [solver_objs['ksp'], solver_params['ksp_rtol'],
                                 solver_params['ksp_atol'], solver_params['ksp_divtol'],
                                 solver_params['ksp_max_it']]
        )

        ksp_set_type = petsc_call(
            'KSPSetType', [solver_objs['ksp'], solver_mapper[solver_params['ksp_type']]]
        )

        ksp_get_pc = petsc_call(
            'KSPGetPC', [solver_objs['ksp'], Byref(solver_objs['pc'])]
        )

        # Even though the default will be jacobi, set to PCNONE for now
        pc_set_type = petsc_call('PCSetType', [solver_objs['pc'], 'PCNONE'])

        ksp_set_from_ops = petsc_call('KSPSetFromOptions', [solver_objs['ksp']])

        matvec_operation = petsc_call(
            'MatShellSetOperation',
            [solver_objs['Jac'], 'MATOP_MULT',
             MatShellSetOp(cbbuilder.matvec_callback.name, void, void)]
        )

        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [solver_objs['snes'], Null,
             FormFunctionCallback(cbbuilder.formfunc_callback.name, void, void), Null]
        )

        dmda_calls = self._create_dmda_calls(dmda, objs)

        mainctx = solver_objs['mainctx']

        call_struct_callback = petsc_call(
            cbbuilder.struct_callback.name, [Byref(mainctx)]
        )
        calls_set_app_ctx = [
            petsc_call('DMSetApplicationContext', [dmda, Byref(mainctx)])
        ]
        calls = [call_struct_callback] + calls_set_app_ctx + [BlankLine]

        base_setup = dmda_calls + (
            snes_create,
            snes_set_dm,
            create_matrix,
            snes_set_jac,
            snes_set_type,
            global_x,
            global_b,
            local_b,
            snes_get_ksp,
            ksp_set_tols,
            ksp_set_type,
            ksp_get_pc,
            pc_set_type,
            ksp_set_from_ops,
            matvec_operation,
            formfunc_operation,
        ) + tuple(calls)

        extended_setup = self._extend_setup(solver_objs, objs, injectsolve, cbbuilder)
        return base_setup + tuple(extended_setup)

    def _extend_setup(self, solver_objs, objs, injectsolve, cbbuilder):
        """
        Hook for subclasses to add additional setup calls.
        """
        return []

    def _create_dmda_calls(self, dmda, objs):
        dmda_create = self._create_dmda(dmda, objs)
        dm_setup = petsc_call('DMSetUp', [dmda])
        dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])
        return dmda_create, dm_setup, dm_mat_type

    def _create_dmda(self, dmda, objs):
        grid = objs['grid']

        nspace_dims = len(grid.dimensions)

        # MPI communicator
        args = [objs['comm']]

        # Type of ghost nodes
        args.extend(['DM_BOUNDARY_GHOSTED' for _ in range(nspace_dims)])

        # Stencil type
        if nspace_dims > 1:
            args.append('DMDA_STENCIL_BOX')

        # Global dimensions
        args.extend(list(grid.shape)[::-1])
        # No.of processors in each dimension
        if nspace_dims > 1:
            args.extend(list(grid.distributor.topology)[::-1])

        # Number of degrees of freedom per node
        args.append(self.dof_per_node)
        # "Stencil width" -> size of overlap
        args.append(dmda.stencil_width)
        args.extend([Null]*nspace_dims)

        # The distributed array object
        args.append(Byref(dmda))

        # The PETSc call used to create the DMDA
        dmda = petsc_call('DMDACreate%sd' % nspace_dims, args)

        return dmda

    @property
    def dof_per_node(self):
        return 1


class CoupledSetup(BaseSetup):

    # TODO: don't actually need to override this, just overriding for purposes of debugging
    #/ engineering the coupled solvers
    def _setup(self, solver_objs, objs, injectsolve, cbbuilder):
        dmda = solver_objs['dmda']

        solver_params = injectsolve.expr.rhs.solver_parameters

        snes_create = petsc_call('SNESCreate', [objs['comm'], Byref(solver_objs['snes'])])

        snes_set_dm = petsc_call('SNESSetDM', [solver_objs['snes'], dmda])

        create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(solver_objs['Jac'])])

        # NOTE: Assuming all solves are linear for now.
        snes_set_type = petsc_call('SNESSetType', [solver_objs['snes'], 'SNESKSPONLY'])

        snes_set_jac = petsc_call(
            'SNESSetJacobian', [solver_objs['snes'], solver_objs['Jac'],
                                solver_objs['Jac'], 'MatMFFDComputeJacobian', Null]
        )

        global_x = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(solver_objs['x_global'])])

        global_b = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(solver_objs['b_global'])])

        local_b = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(solver_objs['b_local'])])

        snes_get_ksp = petsc_call('SNESGetKSP',
                                  [solver_objs['snes'], Byref(solver_objs['ksp'])])

        ksp_set_tols = petsc_call(
            'KSPSetTolerances', [solver_objs['ksp'], solver_params['ksp_rtol'],
                                 solver_params['ksp_atol'], solver_params['ksp_divtol'],
                                 solver_params['ksp_max_it']]
        )

        ksp_set_type = petsc_call(
            'KSPSetType', [solver_objs['ksp'], solver_mapper[solver_params['ksp_type']]]
        )

        ksp_get_pc = petsc_call(
            'KSPGetPC', [solver_objs['ksp'], Byref(solver_objs['pc'])]
        )

        # Even though the default will be jacobi, set to PCNONE for now
        pc_set_type = petsc_call('PCSetType', [solver_objs['pc'], 'PCNONE'])

        ksp_set_from_ops = petsc_call('KSPSetFromOptions', [solver_objs['ksp']])

        matvec_operation = petsc_call(
            'MatShellSetOperation',
            [solver_objs['Jac'], 'MATOP_MULT',
             MatShellSetOp(cbbuilder.matvec_callback.name, void, void)]
        )

        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [solver_objs['snes'], Null,
             FormFunctionCallback(cbbuilder.formfunc_callback.name, void, void), Null]
        )

        dmda_calls = self._create_dmda_calls(dmda, objs)

        mainctx = solver_objs['mainctx']

        call_struct_callback = petsc_call(
            cbbuilder.struct_callback.name, [Byref(mainctx)]
        )
        calls_set_app_ctx = [
            petsc_call('DMSetApplicationContext', [dmda, Byref(mainctx)])
        ]
        calls = [call_struct_callback] + calls_set_app_ctx

        base_setup = dmda_calls + (
            snes_create,
            snes_set_dm,
            create_matrix,
            snes_set_jac,
            snes_set_type,
            global_x,
            global_b,
            local_b,
            snes_get_ksp,
            ksp_set_tols,
            ksp_set_type,
            ksp_get_pc,
            pc_set_type,
            ksp_set_from_ops,
            matvec_operation,
            formfunc_operation,
        ) + tuple(calls)

        extended_setup = self._extend_setup(solver_objs, objs, injectsolve, cbbuilder)
        return base_setup + tuple(extended_setup) + (BlankLine,)

    @property
    def dof_per_node(self):
        return len(self.injectsolve.expr.rhs.fielddata.targets)

    def _extend_setup(self, solver_objs, objs, injectsolve, cbbuilder):
        dmda = solver_objs['dmda']
        create_field_decomp = petsc_call(
            'DMCreateFieldDecomposition',
            [dmda, Null, Null, Byref(solver_objs['fields']), Byref(solver_objs['subdms'])]
            )
        matop_create_submats_op = petsc_call(
            'MatShellSetOperation',
            [solver_objs['Jac'], 'MATOP_CREATE_SUBMATRICES',
             MatShellSetOp(cbbuilder.submatrices_callback.name, void, void)]
        )
        # malloc_whole_jac = petsc_call('PetscMalloc1', [1, solver_objs['jacctx']])
        ffps = [DummyExpr(FieldFromComposite(i._C_symbol, solver_objs['jacctx']), i._C_symbol) for i in solver_objs['jacctx'].fields]
        # ffps
        shell_set_ctx = petsc_call('MatShellSetContext', [solver_objs['Jac'], Byref(solver_objs['jacctx']._C_symbol)])
        
        call_coupled_struct_callback = petsc_call(
            cbbuilder.coupled_struct_callback.name, [Byref(solver_objs['jacctx'])] + solver_objs['jacctx'].fields
        )
        return [create_field_decomp, matop_create_submats_op] + [call_coupled_struct_callback, shell_set_ctx]


class Solver:
    def __init__(self, solver_objs, objs, injectsolve, iters, cbbuilder,
                 timedep=None, **kwargs):
        self.timedep = timedep
        self.calls = self._execute_solve(solver_objs, objs, injectsolve, iters, cbbuilder)
        self.spatial_body = self._spatial_loop_nest(iters, injectsolve)

        space_iter, = self.spatial_body
        self.mapper = {space_iter: self.calls}

    def _execute_solve(self, solver_objs, objs, injectsolve, iters, cbbuilder):
        """
        Assigns the required time iterators to the struct and executes
        the necessary calls to execute the SNES solver.
        """
        struct_assignment = self.timedep.assign_time_iters(solver_objs['mainctx'])

        rhs_callback = cbbuilder.formrhs_callback

        dmda = solver_objs['dmda']

        rhs_call = petsc_call(rhs_callback.name, list(rhs_callback.parameters))

        local_x = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(solver_objs['x_local'])])

        vec_replace_array = self.timedep.replace_array(solver_objs)

        dm_local_to_global_x = petsc_call(
            'DMLocalToGlobal', [dmda, solver_objs['x_local'], 'INSERT_VALUES',
                                solver_objs['x_global']]
        )

        dm_local_to_global_b = petsc_call(
            'DMLocalToGlobal', [dmda, solver_objs['b_local'], 'INSERT_VALUES',
                                solver_objs['b_global']]
        )

        snes_solve = petsc_call('SNESSolve', [
            solver_objs['snes'], solver_objs['b_global'], solver_objs['x_global']]
        )

        dm_global_to_local_x = petsc_call('DMGlobalToLocal', [
            dmda, solver_objs['x_global'], 'INSERT_VALUES', solver_objs['x_local']]
        )

        run_solver_calls = (struct_assignment,) + (
            rhs_call,
            local_x
        ) + vec_replace_array + (
            dm_local_to_global_x,
            dm_local_to_global_b,
            snes_solve,
            dm_global_to_local_x,
            BlankLine,
        )
        return List(body=run_solver_calls)

    def _spatial_loop_nest(self, iters, injectsolve):
        spatial_body = []
        for tree in retrieve_iteration_tree(iters[0]):
            root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
            if injectsolve in FindNodes(InjectSolveDummy).visit(root):
                spatial_body.append(root)
        return spatial_body


class CoupledSolver(Solver):
    # NOTE: note this is obvs for debugging, shouldn't acc need to override this whole function
    def _execute_solve(self, solver_objs, objs, injectsolve, iters, cbbuilder):
        """
        Assigns the required time iterators to the struct and executes
        the necessary calls to execute the SNES solver.
        """
        struct_assignment = self.timedep.assign_time_iters(solver_objs['mainctx'])

        rhs_callback = cbbuilder.formrhs_callback

        dmda = solver_objs['dmda']

        rhs_call = petsc_call(rhs_callback.name, list(rhs_callback.parameters))

        local_x = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(solver_objs['x_local'])])

        vec_replace_array = self.timedep.replace_array(solver_objs)

        dm_local_to_global_x = petsc_call(
            'DMLocalToGlobal', [dmda, solver_objs['x_local'], 'INSERT_VALUES',
                                solver_objs['x_global']]
        )

        dm_local_to_global_b = petsc_call(
            'DMLocalToGlobal', [dmda, solver_objs['b_local'], 'INSERT_VALUES',
                                solver_objs['b_global']]
        )

        snes_solve = petsc_call('SNESSolve', [
            solver_objs['snes'], solver_objs['b_global'], solver_objs['x_global']]
        )

        dm_global_to_local_x = petsc_call('DMGlobalToLocal', [
            dmda, solver_objs['x_global'], 'INSERT_VALUES', solver_objs['x_local']]
        )

        run_solver_calls = (struct_assignment,) + (
            rhs_call,
            local_x,
        ) + vec_replace_array + (
            dm_local_to_global_x,
            dm_local_to_global_b,
            snes_solve,
            dm_global_to_local_x,
            BlankLine,
        )
        return List(body=run_solver_calls)


class NonTimeDependent:
    def __init__(self, injectsolve, iters, **kwargs):
        self.injectsolve = injectsolve
        self.iters = iters
        self.kwargs = kwargs
        self.origin_to_moddim = self._origin_to_moddim_mapper(iters)
        self.time_idx_to_symb = injectsolve.expr.rhs.time_mapper

    # @property
    # def is_target_time(self):
    #     return False

    @property
    # TODO: for coupled solves, could have a case where one function is a TimeFunction
    # but the other is a Function, but they both depend on time.
    def targets(self):
        return self.injectsolve.expr.rhs.fielddata.targets

    def _origin_to_moddim_mapper(self, iters):
        return {}

    def uxreplace_time(self, body):
        return body

    def replace_array(self, solver_objs):
        """
        VecReplaceArray() is a PETSc function that allows replacing the array
        of a `Vec` with a user provided array.
        https://petsc.org/release/manualpages/Vec/VecReplaceArray/

        This function is used to replace the array of the PETSc solution `Vec`
        with the array from the `Function` object representing the target.

        Examples
        --------
        >>> self.target
        f1(x, y)
        >>> call = replace_array(solver_objs)
        >>> print(call)
        PetscCall(VecReplaceArray(x_local_0,f1_vec->data));
        """
        to_replace = []
        for target in self.targets:
            field_from_ptr = FieldFromPointer(
                target.function._C_field_data, target.function._C_symbol
            )
            vec_replace_array = (petsc_call(
                'VecReplaceArray', [solver_objs['x_local'], field_from_ptr]
            ),)
            to_replace.extend(vec_replace_array)
        return tuple(to_replace)

    def assign_time_iters(self, struct):
        return []


class TimeDependent(NonTimeDependent):
    """
    A class for managing time-dependent solvers.

    This includes scenarios where the target is not directly a `TimeFunction`,
    but depends on other functions that are.

    Outline of time loop abstraction with PETSc:

    - At PETScSolve, time indices are replaced with temporary `Symbol` objects
      via a mapper (e.g., {t: tau0, t + dt: tau1}) to prevent the time loop
      from being generated in the callback functions. These callbacks, needed
      for each `SNESSolve` at every time step, don't require the time loop, but
      may still need access to data from other time steps.
    - All `Function` objects are passed through the initial lowering via the
      `LinearSolveExpr` object, ensuring the correct time loop is generated
      in the main kernel.
    - Another mapper is created based on the modulo dimensions
      generated by the `LinearSolveExpr` object in the main kernel
      (e.g., {time: time, t: t0, t + 1: t1}).
    - These two mappers are used to generate a final mapper `symb_to_moddim`
      (e.g. {tau0: t0, tau1: t1}) which is used at the IET level to
      replace the temporary `Symbol` objects in the callback functions with
      the correct modulo dimensions.
    - Modulo dimensions are updated in the matrix context struct at each time
      step and can be accessed in the callback functions where needed.
    """
    # TODO: move these funcs/properties around

    def is_target_time(self, target):
        return any(i.is_Time for i in target.dimensions)

    @property
    def time_spacing(self):
        return self.injectsolve.expr.rhs.grid.stepping_dim.spacing

    def target_time(self, target):
        target_time = [
            i for i, d in zip(target.indices, target.dimensions)
            if d.is_Time
        ]
        assert len(target_time) == 1
        target_time = target_time.pop()
        return target_time

    @property
    def symb_to_moddim(self):
        """
        Maps temporary `Symbol` objects created during `PETScSolve` to their
        corresponding modulo dimensions (e.g. creates {tau0: t0, tau1: t1}).
        """
        mapper = {
            v: k.xreplace({self.time_spacing: 1, -self.time_spacing: -1})
            for k, v in self.time_idx_to_symb.items()
        }
        return {symb: self.origin_to_moddim[mapper[symb]] for symb in mapper}

    def uxreplace_time(self, body):
        return Uxreplace(self.symb_to_moddim).visit(body)

    def _origin_to_moddim_mapper(self, iters):
        """
        Creates a mapper of the origin of the time dimensions to their corresponding
        modulo dimensions from a list of `Iteration` objects.

        Examples
        --------
        >>> iters
        (<WithProperties[affine,sequential]::Iteration time[t0,t1]; (time_m, time_M, 1)>,
         <WithProperties[affine,parallel,parallel=]::Iteration x; (x_m, x_M, 1)>)
        >>> _origin_to_moddim_mapper(iters)
        {time: time, t: t0, t + 1: t1}
        """
        time_iter = [i for i in iters if any(d.is_Time for d in i.dimensions)]
        mapper = {}

        if not time_iter:
            return mapper

        for i in time_iter:
            for d in i.dimensions:
                if d.is_Modulo:
                    mapper[d.origin] = d
                elif d.is_Time:
                    mapper[d] = d
        return mapper

    def replace_array(self, solver_objs):
        """
        In the case that the actual target is time-dependent e.g a `TimeFunction`,
        a pointer to the first element in the array that will be updated during
        the time step is passed to VecReplaceArray().

        Examples
        --------
        >>> self.target
        f1(time + dt, x, y)
        >>> calls = replace_array(solver_objs)
        >>> print(List(body=calls))
        PetscCall(VecGetSize(x_local_0,&(localsize_0)));
        float * start_ptr_0 = (time + 1)*localsize_0 + (float*)(f1_vec->data);
        PetscCall(VecReplaceArray(x_local_0,start_ptr_0));

        >>> self.target
        f1(t + dt, x, y)
        >>> calls = replace_array(solver_objs)
        >>> print(List(body=calls))
        PetscCall(VecGetSize(x_local_0,&(localsize_0)));
        float * start_ptr_0 = t1*localsize_0 + (float*)(f1_vec->data);
        """
        # TODO: improve this
        to_replace = []
        for target in self.targets:
            if self.is_target_time(target):
                mapper = {self.time_spacing: 1, -self.time_spacing: -1}
                target_time = self.target_time(target).xreplace(mapper)

                try:
                    target_time = self.origin_to_moddim[target_time]
                except KeyError:
                    pass

                start_ptr = solver_objs[target.name+'_ptr']
                [target.name+'_ptr']

                vec_get_size = petsc_call(
                    'VecGetSize', [solver_objs['x_local'], Byref(solver_objs['localsize'])]
                )

                field_from_ptr = FieldFromPointer(
                    target.function._C_field_data, target.function._C_symbol
                )

                expr = DummyExpr(
                    start_ptr, cast_mapper[(target.dtype, '*')](field_from_ptr) +
                    Mul(target_time, solver_objs['localsize']), init=True
                )

                vec_replace_array = petsc_call(
                    'VecReplaceArray', [solver_objs['x_local'], start_ptr]
                )
                to_replace.extend([vec_get_size, expr, vec_replace_array])
                # return (vec_get_size, expr, vec_replace_array)
            else:
                tmp = super().replace_array(solver_objs)
                to_replace.extend(tmp)
        return tuple(to_replace)

    def assign_time_iters(self, struct):
        """
        Assign required time iterators to the struct.
        These iterators are updated at each timestep in the main kernel
        for use in callback functions.

        Examples
        --------
        >>> struct
        ctx
        >>> struct.fields
        [h_x, x_M, x_m, f1(t, x), t0, t1]
        >>> assigned = assign_time_iters(struct)
        >>> print(assigned[0])
        ctx.t0 = t0;
        >>> print(assigned[1])
        ctx.t1 = t1;
        """
        to_assign = [
            f for f in struct.fields if (f.is_Dimension and (f.is_Time or f.is_Modulo))
        ]
        time_iter_assignments = [
            DummyExpr(FieldFromComposite(field, struct), field)
            for field in to_assign
        ]
        return time_iter_assignments


Null = Macro('NULL')
void = 'void'
dummyctx = Symbol('lctx')
dummyptr = DummyArg('dummy')


# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
