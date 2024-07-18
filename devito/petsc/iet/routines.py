from collections import OrderedDict

import cgen as c

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody)
from devito.symbolics import Byref, FieldFromPointer, Macro
from devito.petsc.types import PETScStruct
from devito.petsc.iet.nodes import (PETScCallable, FormFunctionCallback,
                                    MatVecCallback)
from devito.petsc.utils import petsc_call


class PETScCallbackBuilder:
    """
    Build IET routines to generate PETSc callback functions.
    """
    def __new__(cls, rcompile=None, **kwargs):
        obj = object.__new__(cls)
        obj.rcompile = rcompile
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
                                      options={'mpi': False})
        body_matvec = self.create_matvec_body(injectsolve, irs_matvec.uiet.body,
                                              solver_objs, objs)

        matvec_callback = PETScCallable(
            'MyMatShellMult_%s' % target.name, body_matvec, retval=objs['err'],
            parameters=(
                solver_objs['Jac'], solver_objs['X_global'], solver_objs['Y_global']
            )
        )

        self._struct_params.extend(irs_matvec.iet.parameters)

        return matvec_callback

    def create_matvec_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

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

        # NOTE: Question: I have placed a chunk of the calls in the `stacks` argument
        # of the `CallableBody` to ensure that these calls precede the `cast` statements.
        # The 'casts' depend on the calls, so this order is necessary. By doing this,
        # I avoid having to manually construct the 'casts' and can allow Devito to handle
        # their construction. Are there any potential issues with this approach?
        body = [body,
                vec_restore_array_y,
                vec_restore_array_x,
                dm_local_to_global_begin,
                dm_local_to_global_end]

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

        matvec_body = CallableBody(
            List(body=body),
            init=tuple([petsc_func_begin_user]),
            stacks=stacks,
            retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])]))

        # Replace data with pointer to data in struct
        subs = {i: FieldFromPointer(i, struct) for i in struct.usr_ctx}
        matvec_body = Uxreplace(subs).visit(matvec_body)

        return matvec_body

    def make_formfunc(self, injectsolve, objs, solver_objs):
        target = injectsolve.expr.rhs.target
        # Compile formfunc `eqns` into an IET via recursive compilation
        irs_formfunc, _ = self.rcompile(injectsolve.expr.rhs.formfuncs,
                                        options={'mpi': False})
        body_formfunc = self.create_formfunc_body(injectsolve, irs_formfunc.uiet.body,
                                                  solver_objs, objs)

        formfunc_callback = PETScCallable(
            'FormFunction_%s' % target.name, body_formfunc, retval=objs['err'],
            parameters=(solver_objs['snes'], solver_objs['X_global'],
                        solver_objs['Y_global']), unused_parameters=(solver_objs['dummy'])
        )
        self._struct_params.extend(irs_formfunc.iet.parameters)

        return formfunc_callback

    def create_formfunc_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

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

        body = [body,
                vec_restore_array_y,
                vec_restore_array_x,
                dm_local_to_global_begin,
                dm_local_to_global_end]

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

        formfunc_body = CallableBody(
            List(body=body),
            init=tuple([petsc_func_begin_user]),
            stacks=stacks,
            retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])]))

        # Replace data with pointer to data in struct
        subs = {i: FieldFromPointer(i, struct) for i in struct.usr_ctx}
        formfunc_body = Uxreplace(subs).visit(formfunc_body)

        return formfunc_body

    def make_formrhs(self, injectsolve, objs, solver_objs):
        target = injectsolve.expr.rhs.target
        # Compile matvec `eqns` into an IET via recursive compilation
        irs_formrhs, _ = self.rcompile(injectsolve.expr.rhs.formrhs,
                                       options={'mpi': False})
        body_formrhs = self.create_formrhs_body(injectsolve, irs_formrhs.uiet.body,
                                                solver_objs, objs)

        formrhs_callback = PETScCallable(
            'FormRHS_%s' % target.name, body_formrhs, retval=objs['err'],
            parameters=(
                solver_objs['Jac'], solver_objs['X_global'], solver_objs['Y_global']
            )
        )
        return formrhs_callback

    def create_formrhs_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

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

        body = [body,
                vec_restore_array]

        stacks = (
            snes_get_dm,
            dm_get_app_context,
            vec_get_array,
            dm_get_local_info,
        )

        formrhs_body = CallableBody(
            List(body=[body]),
            init=tuple([petsc_func_begin_user]),
            stacks=stacks,
            retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])]))

        # Replace data with pointer to data in struct
        subs = {i: FieldFromPointer(i, struct) for i in struct.usr_ctx}
        formrhs_body = Uxreplace(subs).visit(formrhs_body)

        return formrhs_body

    def runsolve(self, solver_objs, objs, rhs_callback, injectsolve):
        target = injectsolve.expr.rhs.target

        dmda = objs['da_so_%s' % target.space_order]

        rhs_call = petsc_call(rhs_callback.name, None)

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

        calls = (rhs_call,
                 dm_local_to_global_x,
                 dm_local_to_global_b,
                 snes_solve,
                 dm_global_to_local_x)

        return calls


def build_petsc_struct(iet, name, liveness):
    # Place all context data required by the shell routines
    # into a PETScStruct
    basics = FindSymbols('basics').visit(iet)
    avoid = FindSymbols('dimensions|indexedbases|writes').visit(iet)
    usr_ctx = [data for data in basics if data not in avoid]
    return PETScStruct(name, usr_ctx, liveness=liveness)


Null = Macro('NULL')
void = 'void'


# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
