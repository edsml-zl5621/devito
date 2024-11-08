from collections import OrderedDict

import cgen as c

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, Callable, BlankLine)
from devito.symbolics import Byref, FieldFromPointer, Macro, cast_mapper
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import ModuloDimension, TimeDimension, Temp, PointerArray, Symbol
from devito.tools import filter_ordered, as_tuple
from devito.petsc.types import PETScArray, FieldDataNest, DummySymb
from devito.petsc.iet.nodes import (PETScCallable, FormFunctionCallback,
                                    MatVecCallback)
from devito.petsc.iet.utils import petsc_call, petsc_struct
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
            'SNESSetJacobian', [solver_objs['snes'], solver_objs['Jac'],
                                solver_objs['Jac'], 'MatMFFDComputeJacobian', Null]
        )
        matvec_operation = petsc_call(
            'MatShellSetOperation', [solver_objs['Jac'], 'MATOP_MULT',
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
            self.sregistry.make_name(prefix='MyMatShellMult_'), body_matvec,
            retval=objs['err'],
            parameters=(
                solver_objs['Jac'], solver_objs['X_global'], solver_objs['Y_global']
            ),
            target = fielddata.target
        )
        return matvec_callback

    def create_matvec_body(self, fielddata, body, solver_objs, objs):
        dmda = fielddata.dmda._rebuild(destroy=False)
        target = fielddata.target

        body = uxreplace_time(body, solver_objs, objs)

        # struct_params = add_struct_params(body)
        struct = DummySymb('dummystruct')
        # struct = objs['struct']._rebuild(liveness='eager', fields=struct_params)

        y_matvec = solver_objs['y_matvec_%s' % target.name]
        x_matvec = solver_objs['x_matvec_%s' % target.name]

        mat_get_dm = petsc_call('MatGetDM', [solver_objs['Jac'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(struct._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['X_local_%s'%target.name])]
        )

        global_to_local = self.dm_global_to_local(dmda, solver_objs, as_tuple(target), prefix='X')

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['Y_local_%s'%target.name])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [solver_objs['Y_local_%s'%target.name], Byref(y_matvec._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [solver_objs['X_local_%s'%target.name], Byref(x_matvec._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(dmda.info)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [solver_objs['Y_local_%s'%target.name], Byref(y_matvec._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [solver_objs['X_local_%s'%target.name], Byref(x_matvec._C_symbol)]
        )

        dm_local_to_global = self.dm_local_to_global(dmda, solver_objs, as_tuple(target), prefix='Y')

        # TODO: Some of the calls are placed in the `stacks` argument of the
        # `CallableBody` to ensure that they precede the `cast` statements. The
        # 'casts' depend on the calls, so this order is necessary. By doing this,
        # you avoid having to manually construct the `casts` and can allow
        # Devito to handle their construction. This is a temporary solution and
        # should be revisited

        body = body._rebuild(
            body=body.body +
            (vec_restore_array_y,
             vec_restore_array_x) + dm_local_to_global
        )

        stacks = (
            mat_get_dm,
            dm_get_app_context,
            dm_get_local_xvec) + global_to_local + (
            dm_get_local_yvec,
            vec_get_array_y,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        # dereference_funcs = [Dereference(i, struct) for i in
        #                      struct.fields if isinstance(i.function, AbstractFunction)]

        matvec_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            # stacks=stacks+tuple(dereference_funcs),
            stacks=stacks,
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # # Replace non-function data with pointer to data in struct
        # subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in struct.fields}
        # matvec_body = Uxreplace(subs).visit(matvec_body)

        # self._struct_params.extend(struct.fields)

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
            target = fielddata.target
        )
        return formfunc_callback

    def create_formfunc_body(self, fielddata, body, solver_objs, objs):
        dmda = fielddata.dmda._rebuild(destroy=False)
        target = fielddata.target

        body = uxreplace_time(body, solver_objs, objs)

        # struct_params = add_struct_params(body)
        # struct = objs['struct']._rebuild(liveness='eager', fields=struct_params)
        struct = DummySymb('dummystruct')

        y_formfunc = solver_objs['y_formfunc_%s'% target.name]
        x_formfunc = solver_objs['x_formfunc_%s'% target.name]

        snes_get_dm = petsc_call('SNESGetDM', [solver_objs['snes'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(struct._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['X_local_%s'% target.name])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, solver_objs['X_global'],
                                     'INSERT_VALUES', solver_objs['X_local_%s'% target.name]]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, solver_objs['X_global'], 'INSERT_VALUES', solver_objs['X_local_%s'% target.name]
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['Y_local_%s'% target.name])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [solver_objs['Y_local_%s'% target.name], Byref(y_formfunc._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [solver_objs['X_local_%s'% target.name], Byref(x_formfunc._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(dmda.info)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [solver_objs['Y_local_%s'% target.name], Byref(y_formfunc._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [solver_objs['X_local_%s'% target.name], Byref(x_formfunc._C_symbol)]
        )

        dm_local_to_global = self.dm_local_to_global(dmda, solver_objs, as_tuple(target), prefix='Y')

        body_new = body._rebuild(
            body=body.body +
            (vec_restore_array_y,
             vec_restore_array_x) + dm_local_to_global
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
        # dereference_funcs = [Dereference(i, struct) for i in
        #                      struct.fields if isinstance(i.function, AbstractFunction)]
        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            # stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )
        
        # Replace non-function data with pointer to data in struct
        # subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in struct.fields}
        # formfunc_body = Uxreplace(subs).visit(formfunc_body)

        # self._struct_params.extend(struct.fields)

        return formfunc_body

    def make_formrhs(self, fielddata, objs, solver_objs):
        target = fielddata.target
        # Compile formrhs `eqns` into an IET via recursive compilation
        irs_formrhs, _ = self.rcompile(fielddata.formrhs,
                                       options={'mpi': False}, sregistry=SymbolRegistry())
        body_formrhs = self.create_formrhs_body(fielddata,
                                                List(body=irs_formrhs.uiet.body),
                                                solver_objs, objs)

        formrhs_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormRHS_'), body_formrhs, retval=objs['err'],
            parameters=(
                solver_objs['snes'], solver_objs['b_local_%s'% target.name],
            ),
            target = fielddata.target
        )

        return formrhs_callback

    def create_formrhs_body(self, fielddata, body, solver_objs, objs):
        dmda = fielddata.dmda._rebuild(destroy=False)
        target = fielddata.target

        snes_get_dm = petsc_call('SNESGetDM', [solver_objs['snes'], Byref(dmda)])

        b_arr = solver_objs['b_tmp_%s'% target.name]

        vec_get_array = petsc_call(
            'VecGetArray', [solver_objs['b_local_%s'% target.name], Byref(b_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(dmda.info)]
        )

        body = uxreplace_time(body, solver_objs, objs)

        # struct_params = add_struct_params(body)
        # struct = objs['struct']._rebuild(liveness='eager', fields=struct_params)
        struct = DummySymb('dummystruct')

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(struct._C_symbol)]
        )

        vec_restore_array = petsc_call(
            'VecRestoreArray', [solver_objs['b_local_%s'% target.name], Byref(b_arr._C_symbol)]
        )

        body = body._rebuild(body=body.body + (vec_restore_array,))

        stacks = (
            snes_get_dm,
            dm_get_app_context,
            vec_get_array,
            dm_get_local_info
        )

        # Dereference function data in struct
        # dereference_funcs = [Dereference(i, struct) for i in
        #                      struct.fields if isinstance(i.function, AbstractFunction)]

        formrhs_body = CallableBody(
            List(body=[body]),
            init=(petsc_func_begin_user,),
            # stacks=stacks+tuple(dereference_funcs),
            stacks=stacks,
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        # subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for
        #         i in struct.fields if not isinstance(i.function, AbstractFunction)}

        # formrhs_body = Uxreplace(subs).visit(formrhs_body)

        # self._struct_params.extend(struct_params)

        return formrhs_body

    def runsolve(self, solver_objs, objs, rhs_callbacks, linsolve):
        dmda = linsolve.parent_dm
        fielddata = linsolve.fielddata
        targets = as_tuple(fielddata.target)
        callbacks = as_tuple(rhs_callbacks)

        rhs_calls = tuple(petsc_call(c.name, list(c.parameters)) for c in callbacks)

        local_x = self.create_local_vecs(dmda, solver_objs, targets, prefix='x')

        vec_replace_array = [i for t in targets for i in self.replace_array(t, solver_objs, objs)]

        dm_local_to_global_x = self.dm_local_to_global(dmda, solver_objs, targets, prefix='x')
        dm_local_to_global_b = self.dm_local_to_global(dmda, solver_objs, targets, prefix='b')

        snes_solve = petsc_call('SNESSolve', [
            solver_objs['snes'], solver_objs['b_global'], solver_objs['x_global']]
        )

        dm_global_to_local_x = self.dm_global_to_local(dmda, solver_objs, targets, prefix='x')

        return rhs_calls + local_x + as_tuple(vec_replace_array) + dm_local_to_global_x + dm_local_to_global_b + (
            snes_solve,) + dm_global_to_local_x 

    def dm_local_to_global(self, dmda, solver_objs, targets, prefix):
        local_to_global = [
            petsc_call('DMLocalToGlobal', [dmda, solver_objs['%s_local_%s'% (prefix, t.name)],
                                           'INSERT_VALUES', solver_objs['%s_global'% prefix]])
            for t in targets
        ]
        return as_tuple(local_to_global)

    def dm_global_to_local(self, dmda, solver_objs, targets, prefix):
        global_to_local = [
            petsc_call('DMGlobalToLocal', [dmda, solver_objs['%s_global'% prefix],
            'INSERT_VALUES', solver_objs['%s_local_%s'% (prefix, t.name)]])
            for t in targets
        ]
        return as_tuple(global_to_local)

    def create_local_vecs(self, dmda, solver_objs, targets, prefix):
        local_vecs = [
            petsc_call('DMCreateLocalVector', [dmda, Byref(solver_objs['%s_local_%s'% (prefix, t.name)])])
            for t in targets
        ]
        return as_tuple(local_vecs)

    def replace_array(self, target, solver_objs, objs):
        if not any(i.is_Time for i in target.dimensions):
            field_from_ptr = FieldFromPointer(target._C_field_data, target._C_symbol)
            vec_replace_array = petsc_call(
                'VecReplaceArray', [solver_objs['x_local_%s'% target.name], field_from_ptr]
            )
            return as_tuple(vec_replace_array)

        target_time = [
            i for i, d in zip(target.indices, target.dimensions) if d.is_Time
        ]
        assert len(target_time) == 1
        target_time = target_time.pop()

        start_ptr = solver_objs['start_ptr_%s' % target.name]

        vec_get_size = petsc_call(
            'VecGetSize', [solver_objs['x_local_%s' % target.name], Byref(solver_objs['localsize_%s' % target.name])]
        )

        field_from_ptr = FieldFromPointer(
            target.function._C_field_data, target.function._C_symbol
        )

        expr = DummyExpr(
            start_ptr, cast_mapper[(target.dtype, '*')](field_from_ptr) +
            Mul(target_time, solver_objs['localsize_%s' % target.name]), init=True
        )

        vec_replace_array = petsc_call('VecReplaceArray', [solver_objs['x_local_%s' % target.name], start_ptr])
        return (vec_get_size, expr, vec_replace_array)


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
            'SNESSetJacobian', [solver_objs['snes'], solver_objs['Jac'],
                                solver_objs['Jac'], FormFunctionCallback(jac_all.name, void, void), Null]
        )

        snes_set_function = petsc_call(
            'SNESSetFunction',
            [solver_objs['snes'], Null, FormFunctionCallback(func_all.name, void, void), Null]
        )

        runsolve = self.runsolve(
            solver_objs, objs, formrhs, linsolve)

        calls = (snes_set_jac, snes_set_function)
        return calls, runsolve

    def form_jacobian_all(self, linsolve, matvec_callbacks, objs, solver_objs):
        parent_dm = linsolve.parent_dm._rebuild(destroy=False)
        children_dms = [dm._rebuild(destroy=False) for dm in linsolve.children_dms]

        targets = linsolve.fielddata.target

        # TODO: only create this one and reuse
        struct = DummySymb('dummystruct')
        # struct = objs['struct']._rebuild(liveness='eager')

        snes_get_dm = petsc_call(
            'SNESGetDM', [solver_objs['snes'], Byref(parent_dm)]
        )
        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [parent_dm, Byref(struct._C_symbol)]
        )
        get_local_vecs = petsc_call(
            'DMCompositeGetLocalVectors',
            [parent_dm] + [Byref(solver_objs['X_local_%s' % t.name]) for t in targets]
        )
        scatter = petsc_call(
            'DMCompositeScatter',
            [parent_dm, solver_objs['X_global']] +  [solver_objs['X_local_%s' % t.name] for t in targets]
        )
        get_local_is = petsc_call(
            'DMCompositeGetLocalIS', [parent_dm, Byref(solver_objs['indexset']._C_symbol)]
        )

        dim = solver_objs['indexset'].dim
        idxset = solver_objs['indexset']
        # TODO: extend to all submatrices not just block diagonal
        # TODO: maybe group all the submatrix calls together in a loop or separate function
        get_local_submats = [petsc_call('MatGetLocalSubMatrix', [solver_objs['Jac'], idxset.indexify().subs({dim: solver_objs['J%s%s' % (t.name, t.name)].row}), idxset.indexify().subs({dim: solver_objs['J%s%s' % (t.name, t.name)].col}), Byref(solver_objs['J%s%s' % (t.name, t.name)])]) for t in targets]

        set_ctx = [petsc_call('MatShellSetContext', [solver_objs['J%s%s' % (t.name, t.name)]] + [struct._C_symbol]) for t in targets]

        matshell_set_op = [
            petsc_call('MatShellSetOperation', [solver_objs['J%s%s' % (t.name, t.name)], 'MATOP_MULT',
                MatVecCallback(next(c.name for c in matvec_callbacks if c.target == t), void, void)
            ])
            for t in targets
        ]
  
        restore_local_submats = [petsc_call('MatRestoreLocalSubMatrix', [solver_objs['Jac'], idxset.indexify().subs({dim: solver_objs['J%s%s' % (t.name, t.name)].row}), idxset.indexify().subs({dim: solver_objs['J%s%s' % (t.name, t.name)].col}), Byref(solver_objs['J%s%s' % (t.name, t.name)])]) for t in targets]
        
        restore_local_vecs = petsc_call(
            'DMCompositeRestoreLocalVectors',
            [parent_dm] + [Byref(solver_objs['X_local_%s' % t.name]) for t in targets]
        )

        mat_assembly_begin = petsc_call('MatAssemblyBegin', [solver_objs['Jac'], 'MAT_FINAL_ASSEMBLY'])
        mat_assembly_end = petsc_call('MatAssemblyEnd', [solver_objs['Jac'], 'MAT_FINAL_ASSEMBLY'])

        # TODO: clean this up
        body = [
            snes_get_dm,
            dm_get_app_context,
            get_local_vecs,
            scatter,
            get_local_is
        ] + get_local_submats + set_ctx + matshell_set_op + restore_local_submats + [restore_local_vecs, mat_assembly_begin, mat_assembly_end]

        body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        jac_all_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormJacobianAll_'), body, retval=objs['err'],
            parameters=(
                solver_objs['snes'], solver_objs['X_global'], solver_objs['Jac'], solver_objs['Jac'], solver_objs['dummy']
            )
        )
        return jac_all_callback

    def form_function_all(self, linsolve, formfunc_callbacks, objs, solver_objs):
        parent_dm = linsolve.parent_dm._rebuild(destroy=False)
        children_dms = [dm._rebuild(destroy=False) for dm in linsolve.children_dms]

        targets = linsolve.fielddata.target

        # TODO: the callback dm doesn't have to be able to take in a body? re-think this
        # struct = objs['struct']._rebuild(liveness='eager')
        struct = DummySymb('dummystruct')

        snes_get_dm = petsc_call(
            'SNESGetDM', [solver_objs['snes'], Byref(parent_dm)]
        )
        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [parent_dm, Byref(struct._C_symbol)]
        )
        get_entries = petsc_call(
            'DMCompositeGetEntries', [parent_dm] + [Byref(dm) for dm in children_dms]
        )
        local_info = [petsc_call('DMDAGetLocalInfo', [dm, Byref(dm.info)]) for dm in children_dms]

        get_local_vecs_x = petsc_call(
            'DMCompositeGetLocalVectors',
            [parent_dm] + [Byref(solver_objs['X_local_%s' % t.name]) for t in targets]
        )
        get_local_vecs_f = petsc_call(
            'DMCompositeGetLocalVectors', [parent_dm] + [Byref(solver_objs['F_local_%s' % t.name]) for t in targets]
        )
        scatter_x = petsc_call(
            'DMCompositeScatter',
            [parent_dm, solver_objs['X_global']] + [Byref(solver_objs['X_local_%s' % t.name]) for t in targets]
        )
        scatter_f = petsc_call(
            'DMCompositeScatter',
            [parent_dm, solver_objs['F_global']] + [Byref(solver_objs['F_local_%s' % t.name]) for t in targets]
        )

        vec_get_array_x = []
        #NOTE: CHANGE ALL THE Y'S TO F'S IN THE FORMFUNCTION CALLBACKS, I THINK CAN LEAVE Y FOR THE MATVECS
        vec_get_array_f = []
        vec_restore_array_x = []
        vec_restore_array_f = []

        for fielddata in linsolve.fielddata.field_data_list:
            name = fielddata.target.name
            x_formfunc = solver_objs['x_formfunc_%s'% name]
            # TODO: CHANGE THESE TO F'S
            y_formfunc = solver_objs['y_formfunc_%s'% name]
            vec_get_array_x.append(petsc_call(
                'VecGetArray', [solver_objs['X_local_%s'% name], Byref(x_formfunc._C_symbol)]
            ))
            vec_get_array_f.append(petsc_call(
                'VecGetArray', [solver_objs['F_local_%s'% name], Byref(y_formfunc._C_symbol)]
            ))
            vec_restore_array_x.append(petsc_call(
                'VecRestoreArray', [solver_objs['X_local_%s'% name], Byref(x_formfunc._C_symbol)]
            ))
            vec_restore_array_f.append(petsc_call(
                'VecRestoreArray', [solver_objs['F_local_%s'% name], Byref(y_formfunc._C_symbol)]
            ))

        formfunc_calls = [petsc_call(callback.name, list(callback.parameters)) for callback in formfunc_callbacks]

        restore_local_vecs_x = petsc_call(
            'DMCompositeRestoreLocalVectors',
            [parent_dm] + [Byref(solver_objs['X_local_%s' % t.name]) for t in targets]
        )
        restore_local_vecs_f = petsc_call(
            'DMCompositeRestoreLocalVectors',
            [parent_dm] + [Byref(solver_objs['F_local_%s' % t.name]) for t in targets]
        )
        
        body = [snes_get_dm, dm_get_app_context, get_entries] + local_info + [get_local_vecs_x, get_local_vecs_f, scatter_x, scatter_f] + vec_get_array_x + vec_get_array_f + formfunc_calls + vec_restore_array_x + vec_restore_array_f + [restore_local_vecs_x, restore_local_vecs_f]

        body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        formfunc_all = PETScCallable(
            self.sregistry.make_name(prefix='FormFunctionAll_'), body, retval=objs['err'],
            parameters=(
                solver_objs['snes'], solver_objs['X_global'], solver_objs['F_global'], solver_objs['dummy']
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

        dummy = DummySymb('dummystruct')
        formfunc_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormFunction_'), body_formfunc,
            retval=objs['err'],
            parameters=(dummy, solver_objs['da_%s' % target.name].info,
                        *(solver_objs['x_formfunc_%s' % t.name]._C_symbol for t in targets),
                        solver_objs['y_formfunc_%s' % target.name]._C_symbol)
        )
        return formfunc_callback

    def create_formfunc_body(self, fielddata, body, solver_objs, objs):
        body = uxreplace_time(body, solver_objs, objs)

        # struct_params = add_struct_params(body)
        # struct = objs['struct']._rebuild(liveness='eager', fields=struct_params)
        struct = DummySymb('dummystruct')

        # Dereference function data in struct
        # dereference_funcs = [Dereference(i, struct) for i in
        #                      struct.fields if isinstance(i.function, AbstractFunction)]
        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )
        
        # Replace non-function data with pointer to data in struct
        # subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in struct.fields}
        # formfunc_body = Uxreplace(subs).visit(formfunc_body)

        # self._struct_params.extend(struct.fields)

        return formfunc_body

    def dm_local_to_global(self, dmda, solver_objs, targets, prefix):
        local_to_global = petsc_call('DMCompositeGather',
        [dmda, 'INSERT_VALUES', solver_objs['%s_global'% prefix]] + [solver_objs['%s_local_%s'% (prefix, t.name)] for t in targets])
        return as_tuple(local_to_global)

    def dm_global_to_local(self, dmda, solver_objs, targets, prefix):
        global_to_local = petsc_call('DMCompositeScatter',
            [dmda, solver_objs['%s_global'% prefix], 'INSERT_VALUES'] + [solver_objs['%s_local_%s'% (prefix, t.name)] for t in targets])
        return as_tuple(global_to_local)

    def create_local_vecs(self, dmda, solver_objs, targets, prefix):
        local_vecs = petsc_call(
            'DMCompositeGetLocalVector', 
            [dmda] + [Byref(solver_objs['%s_local_%s'% (prefix, t.name)]) for t in targets]
        )
        return as_tuple(local_vecs)


def add_struct_params(iet):
    fields = [
        i.function for i in FindSymbols('basics').visit(iet)
        if not isinstance(i.function, (PETScArray, Temp))
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
