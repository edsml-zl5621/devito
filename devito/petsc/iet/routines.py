from collections import OrderedDict

import cgen as c

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, BlankLine)
from devito.symbolics import Byref, FieldFromPointer, Macro, Cast, FieldFromComposite
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import LocalObject, ModuloDimension
from devito.tools import ctypes_to_cstr, dtype_to_ctype, CustomDtype
from devito.petsc.types import PETScArray
from devito.petsc.iet.nodes import (PETScCallable, FormFunctionCallback,
                                    MatVecCallback)
from devito.petsc.iet.utils import petsc_call, petsc_struct, remove_CallbackExpr
from devito.ir.support import SymbolRegistry


class PETScCallbackBuilder:
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
        target = injectsolve.expr.rhs.target
        # Compile matvec `eqns` into an IET via recursive compilation
        irs_matvec, _ = self.rcompile(injectsolve.expr.rhs.matvecs,
                                      options={'mpi': False}, sregistry=SymbolRegistry())
        body_matvec = self.create_matvec_body(injectsolve,
                                              List(body=irs_matvec.uiet.body),
                                              solver_objs, objs)

        matvec_callback = PETScCallable(
            'MyMatShellMult_%s' % target.name, body_matvec, retval=objs['err'],
            parameters=(
                solver_objs['Jac'], solver_objs['X_global'], solver_objs['Y_global']
            )
        )
        return matvec_callback

    def create_matvec_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

        body = remove_CallbackExpr(body)

        dmda = objs['da_so_%s' % linsolveexpr.target.space_order]

        struct = build_petsc_struct(body, 'matvec', liveness='eager')

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
                             struct.fields if isinstance(i.function, AbstractFunction)]

        matvec_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in struct.fields}
        matvec_body = Uxreplace(subs).visit(matvec_body)
        # from IPython import embed; embed()
        self._struct_params.extend(struct.fields)

        return matvec_body

    def make_formfunc(self, injectsolve, objs, solver_objs):
        target = injectsolve.expr.rhs.target
        # Compile formfunc `eqns` into an IET via recursive compilation
        irs_formfunc, _ = self.rcompile(injectsolve.expr.rhs.formfuncs,
                                        options={'mpi': False}, sregistry=SymbolRegistry())
        body_formfunc = self.create_formfunc_body(injectsolve,
                                                  List(body=irs_formfunc.uiet.body),
                                                  solver_objs, objs)

        formfunc_callback = PETScCallable(
            'FormFunction_%s' % target.name, body_formfunc, retval=objs['err'],
            parameters=(solver_objs['snes'], solver_objs['X_global'],
                        solver_objs['Y_global'], solver_objs['dummy'])
        )
        return formfunc_callback

    def create_formfunc_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

        body = remove_CallbackExpr(body)

        dmda = objs['da_so_%s' % linsolveexpr.target.space_order]

        struct = build_petsc_struct(body, 'formfunc', liveness='eager')

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
                             struct.fields if isinstance(i.function, AbstractFunction)]

        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for i in struct.fields}
        formfunc_body = Uxreplace(subs).visit(formfunc_body)

        self._struct_params.extend(struct.fields)

        return formfunc_body

    def make_formrhs(self, injectsolve, objs, solver_objs):
        target = injectsolve.expr.rhs.target
        # Compile formrhs `eqns` into an IET via recursive compilation
        irs_formrhs, _ = self.rcompile(injectsolve.expr.rhs.formrhs,
                                       options={'mpi': False}, sregistry=SymbolRegistry())
        body_formrhs = self.create_formrhs_body(injectsolve,
                                                List(body=irs_formrhs.uiet.body),
                                                solver_objs, objs)

        formrhs_callback = PETScCallable(
            'FormRHS_%s' % target.name, body_formrhs, retval=objs['err'],
            parameters=(
                solver_objs['snes'], solver_objs['b_local']
            )
        )

        return formrhs_callback

    def create_formrhs_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

        body = remove_CallbackExpr(body)

        dmda = objs['da_so_%s' % linsolveexpr.target.space_order]

        snes_get_dm = petsc_call('SNESGetDM', [solver_objs['snes'], Byref(dmda)])

        b_arr = linsolveexpr.arrays['b_tmp']

        vec_get_array = petsc_call(
            'VecGetArray', [solver_objs['b_local'], Byref(b_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(dmda.info)]
        )

        struct = build_petsc_struct(body, 'formrhs', liveness='eager')

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
                             struct.fields if isinstance(i.function, AbstractFunction)]

        formrhs_body = CallableBody(
            List(body=[body]),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for
                i in struct.fields if not isinstance(i.function, AbstractFunction)}
        formrhs_body = Uxreplace(subs).visit(formrhs_body)

        self._struct_params.extend(struct.fields)

        return formrhs_body

    def runsolve(self, solver_objs, objs, rhs_callback, injectsolve):
        target = injectsolve.expr.rhs.target

        # Initialise the modulo dimensions in the main struct 
        if any(i.is_Time for i in injectsolve.expr.dimensions):
            # TODO: set the modulo dims that appear in expr.dimensions but not the one that
            # appears in injectsolve.expr.lhs.indices
            # DummyExpr(FieldFromComposite(sdata.symbolic_flag, sdata[d]), 2)
            if any(isinstance(dim, ModuloDimension) for dim in injectsolve.expr.dimensions):
                # dims_to_initialise = [dim for dim in injectsolve.expr.dimensions
                #                      if dim not in injectsolve.expr.lhs.indices
                #                      and isinstance(dim, ModuloDimension)]
                # dims_to_initialise = [DummyExpr(FieldFromComposite(objs['ctx'], dim), dim) for dim in dims_to_initialise]
                dims_to_initialise = []
            else:
                dims_to_initialise = []
        else:
            dims_to_initialise = []
            # set_modulo_dims = [FieldFromComposite(, threads)]

        dmda = objs['da_so_%s' % target.space_order]

        rhs_call = petsc_call(rhs_callback.name, list(rhs_callback.parameters))

        local_x = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(solver_objs['x_local'])])

        if any(i.is_Time for i in target.dimensions):
            vec_replace_array = time_dep_replace(injectsolve, target, solver_objs, objs)

        else:
            field_from_ptr = FieldFromPointer(target._C_field_data, target._C_symbol)
            vec_replace_array = petsc_call(
                'VecReplaceArray', [solver_objs['x_local'], field_from_ptr]
            )

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

        calls = (dims_to_initialise,
                 rhs_call,
                 local_x,
                 vec_replace_array,
                 dm_local_to_global_x,
                 dm_local_to_global_b,
                 snes_solve,
                 dm_global_to_local_x,
                 BlankLine)

        return calls


def build_petsc_struct(iet, name, liveness):
    # Place all context data required by the shell routines into a struct
    # TODO: Clean this search up
    basics = FindSymbols('basics').visit(iet)
    mod_dims = [i for i in FindSymbols('dimensions').visit(iet)
                if i.is_Modulo]
    avoid0 = [i for i in FindSymbols('indexedbases').visit(iet)
              if isinstance(i.function, PETScArray)]
    avoid1 = [i for i in FindSymbols('dimensions|writes').visit(iet)
              if i not in mod_dims]
    fields = [data.function for data in basics if data not in avoid0+avoid1]
    return petsc_struct(name, fields, liveness)


def time_dep_replace(injectsolve, target, solver_objs, objs):
    target_time = injectsolve.expr.lhs
    target_time = [i for i, d in zip(target_time.indices,
                                     target_time.dimensions) if d.is_Time]
    assert len(target_time) == 1
    target_time = target_time.pop()

    ctype_str = ctypes_to_cstr(dtype_to_ctype(target.dtype))

    class BarCast(Cast):
        _base_typ = ctype_str

    class StartPtr(LocalObject):
        dtype = CustomDtype(ctype_str, modifier=' *')

    start_ptr = StartPtr('start_ptr')

    vec_get_size = petsc_call(
        'VecGetSize', [solver_objs['x_local'], Byref(objs['localsize'])]
    )

    # TODO: What is the correct way to use Mul here? Devito Mul? Sympy Mul?
    field_from_ptr = FieldFromPointer(target._C_field_data, target._C_symbol)
    expr = DummyExpr(
        start_ptr, BarCast(field_from_ptr, ' *') +
        Mul(target_time, objs['localsize']), init=True
    )

    vec_replace_array = petsc_call('VecReplaceArray', [solver_objs['x_local'], start_ptr])
    return [vec_get_size, expr, vec_replace_array]


Null = Macro('NULL')
void = 'void'


# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
