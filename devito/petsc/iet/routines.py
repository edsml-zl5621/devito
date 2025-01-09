from collections import OrderedDict

import cgen as c

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, BlankLine, Callable)
from devito.symbolics import Byref, FieldFromPointer, Macro, cast_mapper
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import ModuloDimension, TimeDimension, Temp
from devito.tools import filter_ordered
from devito.petsc.types import PETScArray
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
        obj.concretize_mapper = kwargs.get('concretize_mapper', {})

        return obj

    @property
    def efuncs(self):
        return self._efuncs

    @property
    def struct_params(self):
        return self._struct_params

    def make(self, injectsolve, objs, solver_objs):
        matvec_callback, formfunc_callback, formrhs_callback = self.make_all(
            injectsolve, objs, solver_objs
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
        runsolve = self.runsolve(solver_objs, objs, formrhs_callback, injectsolve)

        return matvec_operation, formfunc_operation, runsolve

    def make_all(self, injectsolve, objs, solver_objs):
        matvec_callback = self.make_matvec(injectsolve, objs, solver_objs)
        formfunc_callback = self.make_formfunc(injectsolve, objs, solver_objs)
        formrhs_callback = self.make_formrhs(injectsolve, objs, solver_objs)

        self._efuncs[matvec_callback.name] = matvec_callback
        self._efuncs[formfunc_callback.name] = formfunc_callback
        self._efuncs[formrhs_callback.name] = formrhs_callback

        return matvec_callback, formfunc_callback, formrhs_callback

    def make_matvec(self, injectsolve, objs, solver_objs):
        # Compile matvec `eqns` into an IET via recursive compilation
        irs_matvec, _ = self.rcompile(injectsolve.expr.rhs.matvecs,
                                      options={'mpi': False}, sregistry=SymbolRegistry(),
                                      concretize_mapper=self.concretize_mapper)
        body_matvec = self.create_matvec_body(injectsolve,
                                              List(body=irs_matvec.uiet.body),
                                              solver_objs, objs)

        matvec_callback = PETScCallable(
            self.sregistry.make_name(prefix='MyMatShellMult_'), body_matvec,
            retval=objs['err'],
            parameters=(
                solver_objs['Jac'], solver_objs['X_global'], solver_objs['Y_global']
            )
        )
        return matvec_callback

    def create_matvec_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

        dmda = solver_objs['dmda']

        body = uxreplace_time(body, solver_objs)

        struct = solver_objs['localctx']
        fields = dummy_fields(body)

        y_matvec = linsolveexpr.arrays['y_matvec']
        x_matvec = linsolveexpr.arrays['x_matvec']

        mat_get_dm = petsc_call('MatGetDM', [solver_objs['Jac'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(struct._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['X_local'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, solver_objs['X_global'],
                                     'INSERT_VALUES', solver_objs['X_local']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, solver_objs['X_global'], 'INSERT_VALUES', solver_objs['X_local']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['Y_local'])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [solver_objs['Y_local'], Byref(y_matvec._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [solver_objs['X_local'], Byref(x_matvec._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(dmda.info)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [solver_objs['Y_local'], Byref(y_matvec._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [solver_objs['X_local'], Byref(x_matvec._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']
        ])

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
             dm_local_to_global_end)
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
        dereference_funcs = [Dereference(i, struct) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        dereference_targets = [Dereference(linsolveexpr.target, struct)]

        matvec_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs)+tuple(dereference_targets),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in fields}
        matvec_body = Uxreplace(subs).visit(matvec_body)

        self._struct_params.extend(fields)

        return matvec_body

    def make_formfunc(self, injectsolve, objs, solver_objs):
        # Compile formfunc `eqns` into an IET via recursive compilation
        irs_formfunc, _ = self.rcompile(
            injectsolve.expr.rhs.formfuncs,
            options={'mpi': False}, sregistry=SymbolRegistry(),
            concretize_mapper=self.concretize_mapper
        )
        body_formfunc = self.create_formfunc_body(injectsolve,
                                                  List(body=irs_formfunc.uiet.body),
                                                  solver_objs, objs)

        formfunc_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormFunction_'), body_formfunc,
            retval=objs['err'],
            parameters=(solver_objs['snes'], solver_objs['X_global'],
                        solver_objs['Y_global'], solver_objs['dummy'])
        )
        return formfunc_callback

    def create_formfunc_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

        dmda = solver_objs['dmda']

        body = uxreplace_time(body, solver_objs)

        struct = solver_objs['localctx']

        fields = dummy_fields(body)

        y_formfunc = linsolveexpr.arrays['y_formfunc']
        x_formfunc = linsolveexpr.arrays['x_formfunc']

        snes_get_dm = petsc_call('SNESGetDM', [solver_objs['snes'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(struct._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['X_local'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, solver_objs['X_global'],
                                     'INSERT_VALUES', solver_objs['X_local']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, solver_objs['X_global'], 'INSERT_VALUES', solver_objs['X_local']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['Y_local'])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [solver_objs['Y_local'], Byref(y_formfunc._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [solver_objs['X_local'], Byref(x_formfunc._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(dmda.info)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [solver_objs['Y_local'], Byref(y_formfunc._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [solver_objs['X_local'], Byref(x_formfunc._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']
        ])

        body = body._rebuild(
            body=body.body +
            (vec_restore_array_y,
             vec_restore_array_x,
             dm_local_to_global_begin,
             dm_local_to_global_end)
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
        dereference_funcs = [Dereference(i, struct) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        # from IPython import embed; embed()
        dereference_targets = [Dereference(linsolveexpr.target, struct)]

        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs)+tuple(dereference_targets),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in fields}
        formfunc_body = Uxreplace(subs).visit(formfunc_body)

        self._struct_params.extend(fields)

        return formfunc_body

    def make_formrhs(self, injectsolve, objs, solver_objs):
        # Compile formrhs `eqns` into an IET via recursive compilation
        irs_formrhs, _ = self.rcompile(injectsolve.expr.rhs.formrhs,
                                       options={'mpi': False}, sregistry=SymbolRegistry(),
                                       concretize_mapper=self.concretize_mapper)
        body_formrhs = self.create_formrhs_body(injectsolve,
                                                List(body=irs_formrhs.uiet.body),
                                                solver_objs, objs)

        formrhs_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormRHS_'), body_formrhs, retval=objs['err'],
            parameters=(
                solver_objs['snes'], solver_objs['b_local']
            )
        )

        return formrhs_callback

    def create_formrhs_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

        dmda = solver_objs['dmda']

        snes_get_dm = petsc_call('SNESGetDM', [solver_objs['snes'], Byref(dmda)])

        b_arr = linsolveexpr.arrays['b_tmp']

        vec_get_array = petsc_call(
            'VecGetArray', [solver_objs['b_local'], Byref(b_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(dmda.info)]
        )

        body = uxreplace_time(body, solver_objs)

        struct = solver_objs['localctx']
        fields = dummy_fields(body)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(struct._C_symbol)]
        )

        vec_restore_array = petsc_call(
            'VecRestoreArray', [solver_objs['b_local'], Byref(b_arr._C_symbol)]
        )

        body = body._rebuild(body=body.body + (vec_restore_array,))

        stacks = (
            snes_get_dm,
            dm_get_app_context,
            vec_get_array,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, struct) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        dereference_targets = [Dereference(linsolveexpr.target, struct)]
        # from IPython import embed; embed()

        formrhs_body = CallableBody(
            List(body=[body]),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs)+tuple(dereference_targets),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # from IPython import embed; embed()

        # Replace non-function data with pointer to data in struct

        # todo: this can be done on all efuncs i think, not here
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for
                i in fields if not isinstance(i.function, AbstractFunction)}

        formrhs_body = Uxreplace(subs).visit(formrhs_body)

        self._struct_params.extend(fields)

        return formrhs_body

    def runsolve(self, solver_objs, objs, rhs_callback, injectsolve):
        target = injectsolve.expr.rhs.target

        # dmda = objs['da_so_%s' % target.space_order]
        dmda = solver_objs['dmda']

        rhs_call = petsc_call(rhs_callback.name, list(rhs_callback.parameters))

        local_x = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(solver_objs['x_local'])])

        if any(i.is_Time for i in target.dimensions):
            vec_replace_array = time_dep_replace(
                injectsolve, solver_objs, objs, self.sregistry
            )
        else:
            field_from_ptr = FieldFromPointer(target._C_field_data, target._C_symbol)
            vec_replace_array = (petsc_call(
                'VecReplaceArray', [solver_objs['x_local'], field_from_ptr]
            ),)

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

        return (
            rhs_call,
            local_x
        ) + vec_replace_array + (
            dm_local_to_global_x,
            dm_local_to_global_b,
            snes_solve,
            dm_global_to_local_x,
            BlankLine,
        )

    def make_main_struct(self, solver_objs, objs):

        dmda = solver_objs['dmda']

        # from IPython import embed; embed()
        # params = self.struct_params.append(solver_objs['targets'])
        params = filter_ordered(self.struct_params)
        params += [solver_objs['targets']]
        # params = params.append(solver_objs['targets'])
        # from IPython import embed; embed()

        struct_local = petsc_struct(solver_objs['localctx'].name, filter_ordered(params), solver_objs['Jac'].name, liveness='eager')
        struct_main = petsc_struct(self.sregistry.make_name(prefix='ctx'), filter_ordered(params), solver_objs['Jac'].name)

        struct_callback = self.generate_struct_callback(struct_main, objs)
        call_struct_callback = petsc_call(struct_callback.name, [Byref(struct_main)])
        calls_set_app_ctx = [
            petsc_call('DMSetApplicationContext', [dmda, Byref(struct_main)])
        ]
        calls = [call_struct_callback] + calls_set_app_ctx

        self._efuncs[struct_callback.name] = struct_callback
        return struct_local, calls

    def generate_struct_callback(self, struct_main, objs):
        body = [
            DummyExpr(FieldFromPointer(i._C_symbol, struct_main), i._C_symbol)
            for i in struct_main.fields if i not in struct_main.time_dim_fields
        ]
        struct_callback_body = CallableBody(
            List(body=body), init=tuple([petsc_func_begin_user]),
            retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
        )
        struct_callback = Callable(
            self.sregistry.make_name(prefix='PopulateMatContext_'), struct_callback_body, objs['err'],
            parameters=[struct_main]
        )
        return struct_callback

    def uxreplace_efuncs(self, struct, solver_objs):
        efuncs_new = {}
        for key, efunc in self.efuncs.items():
            updated = Uxreplace({solver_objs['localctx']: struct}).visit(
                efunc
            )
            efuncs_new[key] = updated
        return efuncs_new


def build_local_struct(iet, name, liveness):
    # Place all context data required by the shell routines into a struct
    fields = [
        i.function for i in FindSymbols('basics').visit(iet)
        if not isinstance(i.function, (PETScArray, Temp))
        and not (i.is_Dimension and not isinstance(i, (TimeDimension, ModuloDimension)))
    ]
    return petsc_struct(name, fields, liveness)


def dummy_fields(iet):
    # Place all context data required by the shell routines into a struct
    fields = [
        i.function for i in FindSymbols('basics').visit(iet)
        if not isinstance(i.function, (PETScArray, Temp))
        and not (i.is_Dimension and not isinstance(i, (TimeDimension, ModuloDimension)))
    ]
    return fields


def time_dep_replace(injectsolve, solver_objs, objs, sregistry):
    target = injectsolve.expr.lhs
    target_time = [
        i for i, d in zip(target.indices, target.dimensions) if d.is_Time
    ]
    assert len(target_time) == 1
    target_time = target_time.pop()

    start_ptr = solver_objs['start_ptr']

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

    vec_replace_array = petsc_call('VecReplaceArray', [solver_objs['x_local'], start_ptr])
    return (vec_get_size, expr, vec_replace_array)


def uxreplace_time(body, solver_objs):
    # TODO: Potentially introduce a TimeIteration abstraction to simplify
    # all the time processing that is done (searches, replacements, ...)
    # "manually" via free functions
    time_spacing = solver_objs['target'].grid.stepping_dim.spacing
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
