from collections import OrderedDict

import cgen as c

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, BlankLine)
from devito.symbolics import Byref, FieldFromPointer, Macro, cast_mapper
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import ModuloDimension, TimeDimension, Temp
from devito.tools import as_tuple
from devito.petsc.types import PETScArray, PETScObject
from devito.petsc.iet.nodes import (PETScCallable, FormFunctionCallback,
                                    MatVecCallback)
from devito.petsc.iet.utils import petsc_call
from devito.ir.support import SymbolRegistry


class CallbackBuilder:
    """
    Build IET routines to generate PETSc callback functions.
    """
    def __new__(cls, rcompile=None, sregistry=None, **kwargs):
        obj = object.__new__(cls)
        obj.rcompile = rcompile
        obj.sregistry = sregistry
        obj._efuncs = OrderedDict()
        obj._struct_params = []

        return obj

    @property
    def efuncs(self):
        return self._efuncs

    @property
    def struct_params(self):
        return self._struct_params

    def make(self, linsolve, objs, solver_objs):
        matvec_callback, formfunc_callback, formrhs_callback = self.make_all(
            linsolve.fielddata, objs, solver_objs
        )
        snes_set_jac = petsc_call(
            'SNESSetJacobian', [solver_objs['snes'], solver_objs['J'],
                                solver_objs['J'], 'MatMFFDComputeJacobian', Null]
        )
        matvec_operation = petsc_call(
            'MatShellSetOperation', [solver_objs['J'], 'MATOP_MULT',
                                     MatVecCallback(matvec_callback.name, void, void)]
        )
        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [solver_objs['snes'], Null,
             FormFunctionCallback(formfunc_callback.name, void, void), Null]
        )
        runsolve = self.runsolve(
            solver_objs, objs, formrhs_callback, linsolve)

        return (snes_set_jac, matvec_operation, formfunc_operation, BlankLine), runsolve

    def make_all(self, fielddata, objs, solver_objs):
        matvec_callback = self.make_matvec(fielddata, objs, solver_objs)
        formrhs_callback = self.make_formrhs(fielddata, objs, solver_objs)
        formfunc_callback = self.make_formfunc(fielddata, objs, solver_objs)

        self._efuncs[matvec_callback.name] = matvec_callback
        self._efuncs[formfunc_callback.name] = formfunc_callback
        self._efuncs[formrhs_callback.name] = formrhs_callback

        return matvec_callback, formfunc_callback, formrhs_callback

    def make_matvec(self, fielddata, objs, solver_objs):
        # Compile matvec `eqns` into an IET via recursive compilation
        irs_matvec, _ = self.rcompile(fielddata.matvecs,
                                      options={'mpi': False}, sregistry=SymbolRegistry())
        body_matvec = self.create_matvec_body(fielddata,
                                              List(body=irs_matvec.uiet.body),
                                              solver_objs, objs)

        matvec_callback = PETScCallable(
            self.sregistry.make_name(
                prefix='MyMatShellMult_'), body_matvec, retval=objs['err'], parameters=(
                solver_objs['J'], solver_objs['X_global'], solver_objs['Y_global']
            ), target=fielddata.target
        )
        return matvec_callback

    def create_matvec_body(self, fielddata, body, solver_objs, objs):
        dmda = fielddata.dmda._rebuild(destroy=False)
        target = fielddata.target

        body = uxreplace_time(body, solver_objs, objs)

        struct = objs['cbstruct']

        yarr = solver_objs['y_matvec_%s' % target.name]
        xarr = solver_objs['x_matvec_%s' % target.name]

        ylocal = solver_objs['Y_local_%s' % target.name]
        xlocal = solver_objs['X_local_%s' % target.name]

        mat_get_dm = petsc_call('MatGetDM', [solver_objs['J'], Byref(dmda)])

        get_appctx = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(struct._C_symbol)]
        )
        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(xlocal)]
        )
        gtl = self.dm_global_to_local(dmda, solver_objs['X_global'], xlocal)

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(ylocal)]
        )
        get_y = petsc_call('VecGetArray', [ylocal, Byref(yarr._C_symbol)])

        get_x = petsc_call('VecGetArray', [xlocal, Byref(xarr._C_symbol)])

        get_local_info = petsc_call('DMDAGetLocalInfo', [dmda, Byref(dmda.info)])

        yrestore = petsc_call('VecRestoreArray', [ylocal, Byref(yarr._C_symbol)])

        xrestore = petsc_call('VecRestoreArray', [xlocal, Byref(xarr._C_symbol)])

        ltg = self.dm_local_to_global(dmda, ylocal, solver_objs['Y_global'])

        # TODO: Some of the calls are placed in the `stacks` argument of the
        # `CallableBody` to ensure that they precede the `cast` statements. The
        # 'casts' depend on the calls, so this order is necessary. By doing this,
        # you avoid having to manually construct the `casts` and can allow
        # Devito to handle their construction. This is a temporary solution and
        # should be revisited

        body = body._rebuild(body=body.body + (yrestore, xrestore, ltg))

        stacks = (
            mat_get_dm,
            get_appctx,
            dm_get_local_xvec,
            gtl,
            dm_get_local_yvec,
            get_y,
            get_x,
            get_local_info
        )

        # Dereference function data in structs
        params = collect_struct_params(body)
        deref = [
            Dereference(i, struct)
            for i in params if isinstance(i.function, AbstractFunction)
        ]

        matvec_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(deref),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in params}
        matvec_body = Uxreplace(subs).visit(matvec_body)

        self._struct_params.extend(params)
        return matvec_body

    def make_formfunc(self, fielddata, objs, solver_objs):
        # Compile formfunc `eqns` into an IET via recursive compilation
        irs_formfunc, _ = self.rcompile(
            fielddata.formfuncs,
            options={'mpi': False}, sregistry=SymbolRegistry()
        )
        body_formfunc = self.create_formfunc_body(fielddata,
                                                  List(body=irs_formfunc.uiet.body),
                                                  solver_objs, objs)

        formfunc_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormFunction_'), body_formfunc,
            retval=objs['err'],
            parameters=(solver_objs['snes'], solver_objs['X_global'],
                        solver_objs['Y_global'], solver_objs['dummy']),
            target=fielddata.target
        )
        return formfunc_callback

    def create_formfunc_body(self, fielddata, body, solver_objs, objs):
        dmda = fielddata.dmda._rebuild(destroy=False)
        target = fielddata.target

        body = uxreplace_time(body, solver_objs, objs)

        struct = objs['cbstruct']

        yarr = solver_objs['y_formfunc_%s' % target.name]
        xarr = solver_objs['x_formfunc_%s' % target.name]

        ylocal = solver_objs['Y_local_%s' % target.name]
        xlocal = solver_objs['X_local_%s' % target.name]

        snes_get_dm = petsc_call('SNESGetDM', [solver_objs['snes'], Byref(dmda)])

        get_appctx = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(struct._C_symbol)]
        )
        dm_get_local_xvec = petsc_call('DMGetLocalVector', [dmda, Byref(xlocal)])

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin',
            [dmda, solver_objs['X_global'], 'INSERT_VALUES', xlocal]
        )
        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, solver_objs['X_global'], 'INSERT_VALUES', xlocal
        ])
        dm_get_local_yvec = petsc_call('DMGetLocalVector', [dmda, Byref(ylocal)])

        get_y = petsc_call('VecGetArray', [ylocal, Byref(yarr._C_symbol)])

        get_x = petsc_call('VecGetArray', [xlocal, Byref(xarr._C_symbol)])

        dm_get_local_info = petsc_call('DMDAGetLocalInfo', [dmda, Byref(dmda.info)])

        yrestore = petsc_call('VecRestoreArray', [ylocal, Byref(yarr._C_symbol)])

        xrestore = petsc_call('VecRestoreArray', [xlocal, Byref(xarr._C_symbol)])

        ltg = self.dm_local_to_global(dmda, ylocal, solver_objs['Y_global'])

        body = body._rebuild(body=body.body + (yrestore, xrestore, ltg))

        stacks = (
            snes_get_dm,
            get_appctx,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            get_y,
            get_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        params = collect_struct_params(body)
        deref = [
            Dereference(i, struct)
            for i in params if isinstance(i.function, AbstractFunction)
        ]

        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(deref),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in params}
        formfunc_body = Uxreplace(subs).visit(formfunc_body)

        self._struct_params.extend(params)

        return formfunc_body

    def make_formrhs(self, fielddata, objs, solver_objs):
        target = fielddata.target
        # Compile formrhs `eqns` into an IET via recursive compilation
        irs_formrhs, _ = self.rcompile(fielddata.formrhs,
                                       options={'mpi': False}, sregistry=SymbolRegistry())
        body_formrhs = self.create_formrhs_body(fielddata,
                                                List(body=irs_formrhs.uiet.body),
                                                solver_objs, objs)

        dmda = fielddata.dmda
        formrhs_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormRHS_'), body_formrhs, retval=objs['err'],
            parameters=(dmda, solver_objs['b_local_%s' % target.name]),
            target=fielddata.target
        )

        return formrhs_callback

    def create_formrhs_body(self, fielddata, body, solver_objs, objs):
        dm = fielddata.dmda
        target = fielddata.target
        struct = objs['cbstruct']

        get_appctx = petsc_call('DMGetApplicationContext', [dm, Byref(struct._C_symbol)])

        b_arr = solver_objs['b_tmp_%s' % target.name]
        blocal = solver_objs['b_local_%s' % target.name]

        get_b = petsc_call('VecGetArray', [blocal, Byref(b_arr._C_symbol)])

        get_local_info = petsc_call('DMDAGetLocalInfo', [dm, Byref(dm.info)])

        body = uxreplace_time(body, solver_objs, objs)

        vec_restore_array = petsc_call(
            'VecRestoreArray', [blocal, Byref(b_arr._C_symbol)]
        )
        body = body._rebuild(body=body.body + (vec_restore_array,))

        stacks = (
            get_appctx,
            get_b,
            get_local_info
        )

        # Dereference function data in struct
        params = collect_struct_params(body)
        deref = [
            Dereference(i, struct)
            for i in params if isinstance(i.function, AbstractFunction)
        ]

        formrhs_body = CallableBody(
            List(body=[body]),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(deref),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for
                i in params if not isinstance(i.function, AbstractFunction)}

        formrhs_body = Uxreplace(subs).visit(formrhs_body)

        self._struct_params.extend(params)
        return formrhs_body

    def runsolve(self, solver_objs, objs, rhs_callback, linsolve):
        dm = linsolve.parent_dm
        fielddata = linsolve.fielddata
        t = fielddata.target

        xlocal = solver_objs['x_local_%s' % t.name]
        blocal = solver_objs['b_local_%s' % t.name]

        xglobal = solver_objs['x_global']
        bglobal = solver_objs['b_global']

        rhs_call = petsc_call(rhs_callback.name, rhs_callback.parameters)

        create_xlocal = self.create_local_vec(dm, xlocal)

        # TODO: clean this up
        replace_arr = self.replace_array(xlocal, linsolve, t, solver_objs, objs)

        ltgx = self.dm_local_to_global(dm, xlocal, xglobal)
        ltgb = self.dm_local_to_global(dm, blocal, bglobal)

        solve = petsc_call('SNESSolve', [solver_objs['snes'], bglobal, xglobal])

        gtlx = self.dm_global_to_local(dm, xglobal, xlocal)

        return (rhs_call, create_xlocal) + replace_arr + (ltgx, ltgb, solve, gtlx)

    # TODO: these could be moved outside of the builder?
    def dm_local_to_global(self, dmda, lobj, gobj):
        local_to_global = petsc_call(
            'DMLocalToGlobal', [dmda, lobj, 'INSERT_VALUES', gobj]
        )
        return local_to_global

    def dm_global_to_local(self, dmda, gobj, lobj):
        global_to_local = petsc_call(
            'DMGlobalToLocal', [dmda, gobj, 'INSERT_VALUES', lobj]
        )
        return global_to_local

    def create_local_vec(self, dmda, lobj):
        local_vec = petsc_call('DMCreateLocalVector', [dmda, Byref(lobj)])
        return local_vec

    def replace_array(self, lobj, linsolve, target, solverobjs, objs):
        if not any(i.is_Time for i in target.dimensions):
            field_from_ptr = FieldFromPointer(target._C_field_data, target._C_symbol)
            replace_arr = petsc_call('VecReplaceArray', [lobj, field_from_ptr])
            return as_tuple(replace_arr)

        startptr = solverobjs['start_ptr_%s' % target.name]
        localsize = solverobjs['localsize_%s' % target.name]

        target_time = target.indexify().indices[target.time_dim]
        target_time = target_time.subs(solverobjs['true_dims'])

        vec_get_size = petsc_call(
            'VecGetSize', [lobj, Byref(localsize)]
        )
        field_from_ptr = FieldFromPointer(
            target.function._C_field_data, target.function._C_symbol
        )
        expr = DummyExpr(
            startptr, cast_mapper[(target.dtype, '*')](field_from_ptr) +
            Mul(target_time, localsize), init=True
        )
        expr = uxreplace_time(expr, solverobjs, objs)
        replace_arr = petsc_call('VecReplaceArray', [lobj, startptr])

        return (vec_get_size, expr, replace_arr)


class NestedCallbackBuilder(CallbackBuilder):
    """
    Build IET routines to generate PETSc callback functions,
    specifically for MATNEST matrices.
    """
    def make(self, linsolve, objs, solver_objs):
        matvecs, formfuncs, formrhs = [], [], []
        all_fielddata = linsolve.fielddata.field_data_list
        for fielddata in all_fielddata:
            matvec, formfunc, rhs = self.make_all(
                fielddata, objs, solver_objs
            )
            matvecs.append(matvec)
            formfuncs.append(formfunc)
            formrhs.append(rhs)

        jac_all = self.form_jacobian_all(linsolve, matvecs, objs, solver_objs)
        func_all = self.form_function_all(linsolve, formfuncs, objs, solver_objs)

        self._efuncs.update({jac_all.name: jac_all, func_all.name: func_all})

        # CREATE A FORMJACCALLBACK
        snes_set_jac = petsc_call(
            'SNESSetJacobian', [
                solver_objs['snes'], solver_objs['J'], solver_objs['J'],
                FormFunctionCallback(jac_all.name, void, void), Null
            ]
        )
        snes_set_function = petsc_call('SNESSetFunction', [
            solver_objs['snes'], Null, FormFunctionCallback(func_all.name, void, void),
            Null
        ])
        runsolve = self.runsolve(
            solver_objs, objs, formrhs, linsolve)

        calls = (snes_set_jac, snes_set_function)
        return calls, runsolve

    def form_jacobian_all(self, linsolve, matvecs, objs, solver_objs):
        dm = linsolve.parent_dm._rebuild(destroy=False)
        targets = linsolve.fielddata.targets
        struct = objs['cbstruct']

        snes_get_dm = petsc_call(
            'SNESGetDM', [solver_objs['snes'], Byref(dm)]
        )
        get_appctx = petsc_call(
            'DMGetApplicationContext', [dm, Byref(struct._C_symbol)]
        )
        get_local_vecs = petsc_call(
            'DMCompositeGetLocalVectors',
            [dm] + [Byref(solver_objs['X_local_%s' % t.name]) for t in targets]
        )
        # TODO: helper funcs?
        scatter = petsc_call(
            'DMCompositeScatter', [dm, solver_objs['X_global']] + [
                solver_objs['X_local_%s' % t.name] for t in targets
            ]
        )
        get_local_is = petsc_call(
            'DMCompositeGetLocalISs', [dm, Byref(solver_objs['indexset']._C_symbol)]
        )
        # TODO: extend to all submatrices not just block diagonal and maybe
        # group all the submatrix calls together in a loop or separate function
        idxs = solver_objs['indexset']
        dim = idxs.dim
        get_local_submats = []
        restore_local_submats = []
        for t in targets:
            row = idxs.indexify().subs({dim: solver_objs['B%s%s' % (t.name, t.name)].row})
            col = idxs.indexify().subs({dim: solver_objs['B%s%s' % (t.name, t.name)].col})
            diagmat = solver_objs['B%s%s' % (t.name, t.name)]

            get_local_submats.append(petsc_call(
                'MatGetLocalSubMatrix', [solver_objs['B'], row, col, Byref(diagmat)]
            ))
            restore_local_submats.append(petsc_call(
                'MatRestoreLocalSubMatrix', [solver_objs['B'], row, col, Byref(diagmat)]
            ))

        matshell_set_op = [
            petsc_call(
                'MatShellSetOperation', [
                    solver_objs['B%s%s' % (t.name, t.name)], 'MATOP_MULT',
                    MatVecCallback(
                        next(c.name for c in matvecs if c.target == t), void, void
                    )
                ]
            ) for t in targets
        ]
        restore_local_vecs = petsc_call(
            'DMCompositeRestoreLocalVectors',
            [dm] + [Byref(solver_objs['X_local_%s' % t.name]) for t in targets]
        )
        mat_assembly_begin = petsc_call(
            'MatAssemblyBegin', [solver_objs['B'], 'MAT_FINAL_ASSEMBLY']
        )
        mat_assembly_end = petsc_call(
            'MatAssemblyEnd', [solver_objs['B'], 'MAT_FINAL_ASSEMBLY']
        )

        # TODO: clean this up
        body = [
            snes_get_dm,
            get_appctx,
            get_local_vecs,
            scatter,
            get_local_is
        ] + get_local_submats + matshell_set_op + restore_local_submats + [
            restore_local_vecs, mat_assembly_begin, mat_assembly_end
        ]

        body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )
        jac_all_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormJacobianAll_'), body, retval=objs['err'],
            parameters=(
                solver_objs['snes'], solver_objs['X_global'], solver_objs['J'],
                solver_objs['B'], solver_objs['dummy']
            )
        )
        return jac_all_callback

    def form_function_all(self, linsolve, formfuncs, objs, solver_objs):
        parent_dm = linsolve.parent_dm._rebuild(destroy=False)
        children_dms = [dm._rebuild(destroy=False) for dm in linsolve.children_dms]
        targets = linsolve.fielddata.targets
        struct = objs['cbstruct']

        snes_get_dm = petsc_call(
            'SNESGetDM', [solver_objs['snes'], Byref(parent_dm)]
        )
        get_appctx = petsc_call(
            'DMGetApplicationContext', [parent_dm, Byref(struct._C_symbol)]
        )
        get_entries = petsc_call(
            'DMCompositeGetEntries', [parent_dm] + [Byref(dm) for dm in children_dms]
        )
        local_info = [
            petsc_call('DMDAGetLocalInfo', [dm, Byref(dm.info)]) for dm in children_dms
        ]
        get_local_vecs_x = petsc_call(
            'DMCompositeGetLocalVectors',
            [parent_dm] + [Byref(solver_objs['X_local_%s' % t.name]) for t in targets]
        )
        get_local_vecs_f = petsc_call(
            'DMCompositeGetLocalVectors', [parent_dm] + [
                Byref(solver_objs['F_local_%s' % t.name]) for t in targets
            ]
        )
        scatter_x = petsc_call(
            'DMCompositeScatter', [parent_dm, solver_objs['X_global']] + [
                solver_objs['X_local_%s' % t.name] for t in targets
            ]
        )
        scatter_f = petsc_call(
            'DMCompositeScatter', [parent_dm, solver_objs['F_global']] + [
                solver_objs['F_local_%s' % t.name] for t in targets
            ]
        )

        get_x = []
        # NOTE: CHANGE ALL THE Y'S TO F'S IN THE FORMFUNCTION CALLBACKS,
        # I THINK CAN LEAVE Y FOR THE MATVECS
        get_f = []
        restore_xarr = []
        restore_farr = []

        for fielddata in linsolve.fielddata.field_data_list:
            name = fielddata.target.name

            xarr = solver_objs['x_formfunc_%s' % name]
            yarr = solver_objs['y_formfunc_%s' % name]  # TODO: CHANGE THESE TO F'S
            xlocal = solver_objs['X_local_%s' % name]
            flocal = solver_objs['F_local_%s' % name]

            get_x.append(petsc_call(
                'VecGetArray', [xlocal, Byref(xarr._C_symbol)]
            ))
            get_f.append(petsc_call(
                'VecGetArray', [flocal, Byref(yarr._C_symbol)]
            ))
            restore_xarr.append(petsc_call(
                'VecRestoreArray', [xlocal, Byref(xarr._C_symbol)]
            ))
            restore_farr.append(petsc_call(
                'VecRestoreArray', [flocal, Byref(yarr._C_symbol)]
            ))

        formfunc_calls = [petsc_call(c.name, list(c.parameters)) for c in formfuncs]

        restore_xvecs = petsc_call(
            'DMCompositeRestoreLocalVectors',
            [parent_dm] + [Byref(solver_objs['X_local_%s' % t.name]) for t in targets]
        )
        restore_fvecs = petsc_call(
            'DMCompositeRestoreLocalVectors',
            [parent_dm] + [Byref(solver_objs['F_local_%s' % t.name]) for t in targets]
        )

        body = [
            snes_get_dm,
            get_appctx,
            get_entries
        ] + local_info + [
            get_local_vecs_x,
            get_local_vecs_f,
            scatter_x,
            scatter_f
        ] + get_x + get_f + formfunc_calls + restore_xarr + restore_farr + [
            restore_xvecs, restore_fvecs
        ]

        body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        formfunc_all = PETScCallable(
            self.sregistry.make_name(prefix='FormFunctionAll_'), body, retval=objs['err'],
            parameters=(
                solver_objs['snes'], solver_objs['X_global'], solver_objs['F_global'],
                solver_objs['dummy']
            )
        )
        return formfunc_all

    def make_formfunc(self, fielddata, objs, solver_objs):
        target = fielddata.target
        targets = solver_objs['targets']
        # Compile formfunc `eqns` into an IET via recursive compilation
        irs_formfunc, _ = self.rcompile(
            fielddata.formfuncs,
            options={'mpi': False}, sregistry=SymbolRegistry()
        )

        body_formfunc = self.create_formfunc_body(fielddata,
                                                  List(body=irs_formfunc.uiet.body),
                                                  solver_objs, objs)

        struct = objs['cbstruct']
        formfunc_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormFunction_'), body_formfunc,
            retval=objs['err'],
            parameters=(
                struct, solver_objs['da_%s' % target.name].info,
                *(solver_objs['x_formfunc_%s' % t.name]._C_symbol for t in targets),
                solver_objs['y_formfunc_%s' % target.name]._C_symbol
            )
        )
        return formfunc_callback

    def create_formfunc_body(self, fielddata, body, solver_objs, objs):
        body = uxreplace_time(body, solver_objs, objs)
        struct = objs['cbstruct']

        # Dereference function data in struct
        params = collect_struct_params(body)
        deref = [
            Dereference(i, struct)
            for i in params if isinstance(i.function, AbstractFunction)
        ]

        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=tuple(deref),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in params}
        formfunc_body = Uxreplace(subs).visit(formfunc_body)

        self._struct_params.extend(params)
        return formfunc_body

    def runsolve(self, solver_objs, objs, rhs_callbacks, linsolve):
        dmda = linsolve.parent_dm
        nestdata = linsolve.fielddata
        targets = nestdata.targets

        rhs_calls = as_tuple(petsc_call(c.name, c.parameters) for c in rhs_callbacks)

        xglobal = solver_objs['x_global']
        bglobal = solver_objs['b_global']

        local_vecs = petsc_call(
            'DMCompositeGetLocalVectors',
            [dmda] + [Byref(solver_objs['x_local_%s' % t.name]) for t in targets]
        )
        replace = [
            item for t in targets
            for item in self.replace_array(
                solver_objs['x_local_%s' % t.name],
                linsolve, t, solver_objs, objs
            )
        ]
        xgather = petsc_call(
            'DMCompositeGather', [dmda, 'INSERT_VALUES', xglobal] + [
                solver_objs['x_local_%s' % t.name] for t in targets
            ]
        )
        bgather = petsc_call(
            'DMCompositeGather', [dmda, 'INSERT_VALUES', bglobal] + [
                solver_objs['b_local_%s' % t.name] for t in targets
            ]
        )
        solve = petsc_call('SNESSolve', [solver_objs['snes'], bglobal, xglobal])

        scatter = petsc_call(
            'DMCompositeScatter',
            [dmda, xglobal] + [solver_objs['x_local_%s' % t.name] for t in targets]
        )

        gather_solve_scatter = (xgather, bgather, solve, scatter)
        return rhs_calls + as_tuple(local_vecs) + as_tuple(replace) + gather_solve_scatter


def collect_struct_params(iet):
    fields = [
        i.function for i in FindSymbols('basics').visit(iet)
        if not isinstance(i.function, (PETScArray, Temp, PETScObject))
        and not (i.is_Dimension and not isinstance(i, (TimeDimension, ModuloDimension)))
    ]
    return fields


def uxreplace_time(body, solver_objs, objs):
    # TODO: Potentially introduce a TimeIteration abstraction to simplify
    # all the time processing that is done (searches, replacements, ...)
    # "manually" via free functions
    time_spacing = objs['grid'].stepping_dim.spacing
    true_dims = solver_objs['true_dims']
    time_mapper = {
        v: k.xreplace({time_spacing: 1, -time_spacing: -1})
        for k, v in solver_objs['time_mapper'].items()
    }
    subs = {symb: true_dims[time_mapper[symb]] for symb in time_mapper}
    return Uxreplace(subs).visit(body)


Null = Macro('NULL')
void = 'void'


# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
