from collections import OrderedDict

import cgen as c

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, BlankLine, Callable, FindNodes,
                           Iteration)
from devito.symbolics import (Byref, FieldFromPointer, Macro, cast_mapper,
                              FieldFromComposite)
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import ModuloDimension, TimeDimension, Temp, Symbol
from devito.tools import filter_ordered
from devito.ir.support import SymbolRegistry

from devito.petsc.types import PETScArray
from devito.petsc.iet.nodes import (PETScCallable, FormFunctionCallback,
                                    MatVecCallback)
from devito.petsc.iet.utils import petsc_call, petsc_struct
from devito.petsc.utils import solver_mapper
from devito.petsc.types import (DM, CallbackDM, Mat, LocalVec, GlobalVec, KSP, PC,
                                SNES, DummyArg, PetscInt, StartPtr)


class CallbackBuilder:
    """
    Build IET routines to generate PETSc callback functions.
    """
    def __new__(cls, rcompile=None, sregistry=None, dep=None, **kwargs):
        obj = object.__new__(cls)
        obj.rcompile = rcompile
        obj.sregistry = sregistry
        obj.concretize_mapper = kwargs.get('concretize_mapper', {})
        obj.dep = dep

        obj._efuncs = OrderedDict()
        obj._struct_params = []

        obj._matvec_callback = None
        obj._formfunc_callback = None
        obj._formrhs_callback = None
        obj._struct_callback = None

        return obj

    @property
    def efuncs(self):
        return self._efuncs

    @property
    def struct_params(self):
        return self._struct_params

    @property
    def matvec_callback(self):
        return self._matvec_callback

    @property
    def formfunc_callback(self):
        return self._formfunc_callback

    @property
    def formrhs_callback(self):
        return self._formrhs_callback

    @property
    def struct_callback(self):
        return self._struct_callback

    def make_core(self, injectsolve, objs, solver_objs):
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
        self._matvec_callback = matvec_callback
        return matvec_callback

    def create_matvec_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

        dmda = solver_objs['callbackdm']

        body = self.dep.uxreplace_time(body, solver_objs)

        struct = solver_objs['dummyctx']
        fields = self.dummy_fields(body, solver_objs)

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
            'DMDAGetLocalInfo', [dmda, Byref(linsolveexpr.localinfo)]
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

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(solver_objs['X_local'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(solver_objs['Y_local'])]
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
        dereference_funcs = [Dereference(i, struct) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        matvec_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
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
        self._formfunc_callback = formfunc_callback
        return formfunc_callback

    def create_formfunc_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

        dmda = solver_objs['callbackdm']

        body = self.dep.uxreplace_time(body, solver_objs)

        struct = solver_objs['dummyctx']

        fields = self.dummy_fields(body, solver_objs)

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
            'DMDAGetLocalInfo', [dmda, Byref(linsolveexpr.localinfo)]
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

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(solver_objs['X_local'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(solver_objs['Y_local'])]
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
        dereference_funcs = [Dereference(i, struct) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
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
        self._formrhs_callback = formrhs_callback
        return formrhs_callback

    def create_formrhs_body(self, injectsolve, body, solver_objs, objs):
        linsolveexpr = injectsolve.expr.rhs

        dmda = solver_objs['callbackdm']

        snes_get_dm = petsc_call('SNESGetDM', [solver_objs['snes'], Byref(dmda)])

        b_arr = linsolveexpr.arrays['b_tmp']

        vec_get_array = petsc_call(
            'VecGetArray', [solver_objs['b_local'], Byref(b_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolveexpr.localinfo)]
        )

        body = self.dep.uxreplace_time(body, solver_objs)

        struct = solver_objs['dummyctx']
        fields = self.dummy_fields(body, solver_objs)

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

        formrhs_body = CallableBody(
            List(body=[body]),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, struct) for
                i in fields if not isinstance(i.function, AbstractFunction)}
        formrhs_body = Uxreplace(subs).visit(formrhs_body)

        self._struct_params.extend(fields)

        return formrhs_body

    def local_struct(self, solver_objs):
        """
        This is the struct used within callback functions,
        usually accessed via DMGetApplicationContext.
        """
        params = filter_ordered(self.struct_params)

        return petsc_struct(
            solver_objs['dummyctx'].name,
            filter_ordered(params),
            solver_objs['Jac'].name+'_ctx',
            liveness='eager'
        )

    def main_struct(self, solver_objs):
        """
        This is the struct initialised inside the main kernel and attached to the DM via
        DMSetApplicationContext
        """
        params = filter_ordered(self.struct_params)

        return petsc_struct(
            self.sregistry.make_name(prefix='ctx'),
            filter_ordered(params),
            solver_objs['Jac'].name+'_ctx'
        )

    def make_struct_callback(self, solver_objs, objs):
        struct_main = solver_objs['mainctx']

        body = [
            DummyExpr(FieldFromPointer(i._C_symbol, struct_main), i._C_symbol)
            for i in struct_main.fields if i not in struct_main.time_dim_fields
        ]
        struct_callback_body = CallableBody(
            List(body=body), init=tuple([petsc_func_begin_user]),
            retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
        )
        struct_callback = Callable(
            self.sregistry.make_name(prefix='PopulateMatContext_'),
            struct_callback_body, objs['err'],
            parameters=[struct_main]
        )
        self._efuncs[struct_callback.name] = struct_callback
        self._struct_callback = struct_callback
        return struct_callback

    def dummy_fields(self, iet, solver_objs):
        # Place all context data required by the shell routines into a struct
        fields = [
            i.function for i in FindSymbols('basics').visit(iet)
            if not isinstance(i.function, (PETScArray, Temp))
            and not (
                i.is_Dimension and not isinstance(i, (TimeDimension, ModuloDimension))
            )
        ]
        fields = filter_ordered(fields)
        return fields


class ObjectBuilder:
    """
    A base class for constructing objects needed for a PETSc solver.
    Designed to be extended by subclasses, which can override the `build`
    method to support specific use cases.
    """
    def __new__(cls, sregistry=None, dep=None, **kwargs):
        obj = object.__new__(cls)
        obj.sregistry = sregistry
        obj.dep = dep
        return obj

    def build(self, injectsolve, iters):
        target = injectsolve.expr.rhs.target
        sreg = self.sregistry
        return {
            'Jac': Mat(sreg.make_name(prefix='J_')),
            'x_global': GlobalVec(sreg.make_name(prefix='x_global_')),
            'x_local': LocalVec(sreg.make_name(prefix='x_local_'), liveness='eager'),
            'b_global': GlobalVec(sreg.make_name(prefix='b_global_')),
            'b_local': LocalVec(sreg.make_name(prefix='b_local_')),
            'ksp': KSP(sreg.make_name(prefix='ksp_')),
            'pc': PC(sreg.make_name(prefix='pc_')),
            'snes': SNES(sreg.make_name(prefix='snes_')),
            'X_global': GlobalVec(sreg.make_name(prefix='X_global_')),
            'Y_global': GlobalVec(sreg.make_name(prefix='Y_global_')),
            'X_local': LocalVec(sreg.make_name(prefix='X_local_'), liveness='eager'),
            'Y_local': LocalVec(sreg.make_name(prefix='Y_local_'), liveness='eager'),
            'dummy': DummyArg(sreg.make_name(prefix='dummy_')),
            'localsize': PetscInt(sreg.make_name(prefix='localsize_')),
            'start_ptr': StartPtr(sreg.make_name(prefix='start_ptr_'), target.dtype),
            'true_dims': self.dep.retrieve_time_dims(iters),
            # TODO: extend to targets
            'target': target,
            'time_mapper': injectsolve.expr.rhs.time_mapper,
            'dmda': DM(sreg.make_name(prefix='da_'), liveness='eager',
                       stencil_width=target.space_order),
            'callbackdm': CallbackDM(sreg.make_name(prefix='callbackda_'),
                                     liveness='eager', stencil_width=target.space_order),
            'dummyctx': Symbol('lctx')
        }


class SetupSolver:
    def setup(self, solver_objs, objs, injectsolve, builder):
        dmda = solver_objs['dmda']

        solver_params = injectsolve.expr.rhs.solver_parameters

        snes_create = petsc_call('SNESCreate', [objs['comm'], Byref(solver_objs['snes'])])

        snes_set_dm = petsc_call('SNESSetDM', [solver_objs['snes'], dmda])

        create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(solver_objs['Jac'])])

        # NOTE: Assumming all solves are linear for now.
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
             MatVecCallback(builder.matvec_callback.name, void, void)]
        )

        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [solver_objs['snes'], Null,
             FormFunctionCallback(builder.formfunc_callback.name, void, void), Null]
        )

        dmda_calls = self.create_dmda_calls(dmda, objs)

        mainctx = solver_objs['mainctx']

        call_struct_callback = petsc_call(builder.struct_callback.name, [Byref(mainctx)])
        calls_set_app_ctx = [
            petsc_call('DMSetApplicationContext', [dmda, Byref(mainctx)])
        ]
        calls = [call_struct_callback] + calls_set_app_ctx

        return dmda_calls + (
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

    def create_dmda_calls(self, dmda, objs):
        dmda_create = self.create_dmda(dmda, objs)
        dm_setup = petsc_call('DMSetUp', [dmda])
        dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])
        return dmda_create, dm_setup, dm_mat_type

    def create_dmda(self, dmda, objs):
        no_of_space_dims = len(objs['grid'].dimensions)

        # MPI communicator
        args = [objs['comm']]

        # Type of ghost nodes
        args.extend(['DM_BOUNDARY_GHOSTED' for _ in range(no_of_space_dims)])

        # Stencil type
        if no_of_space_dims > 1:
            args.append('DMDA_STENCIL_BOX')

        # Global dimensions
        args.extend(list(objs['grid'].shape)[::-1])
        # No.of processors in each dimension
        if no_of_space_dims > 1:
            args.extend(list(objs['grid'].distributor.topology)[::-1])

        # Number of degrees of freedom per node
        args.append(1)
        # "Stencil width" -> size of overlap
        args.append(dmda.stencil_width)
        args.extend([Null for _ in range(no_of_space_dims)])

        # The distributed array object
        args.append(Byref(dmda))

        # The PETSc call used to create the DMDA
        dmda = petsc_call('DMDACreate%sd' % no_of_space_dims, args)

        return dmda


class RunSolver:
    def __new__(cls, dep=None, **kwargs):
        obj = object.__new__(cls)
        obj.dep = dep
        return obj

    def run(self, solver_objs, objs, injectsolve, iters, cbbuilder):
        """
        Returns a mapper, mapping the spatial loop nest to the calls
        to run the SNES solver.
        """
        time_iters = self.dep.assign_time_iters(iters, solver_objs['mainctx'])
        runsolve = self.runsolve(
            solver_objs, objs, cbbuilder.formrhs_callback, injectsolve
        )
        calls = List(body=tuple(time_iters)+runsolve)
        return calls

    def runsolve(self, solver_objs, objs, rhs_callback, injectsolve):
        dmda = solver_objs['dmda']

        rhs_call = petsc_call(rhs_callback.name, list(rhs_callback.parameters))

        local_x = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(solver_objs['x_local'])])

        vec_replace_array = self.dep.time_dep_replace(injectsolve, solver_objs, objs)

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


class NonTimeDependent:
    def __new__(cls, injectsolve, **kwargs):
        obj = object.__new__(cls)
        obj.injectsolve = injectsolve
        return obj

    @property
    def is_target_time(self):
        return False

    @property
    def target(self):
        return self.injectsolve.expr.rhs.target

    def uxreplace_time(self, body, solver_objs):
        return body

    def retrieve_time_dims(self, iters):
        return {}

    def time_dep_replace(self, injectsolve, solver_objs, objs):
        return ()

    def assign_time_iters(self, iters, struct):
        return []


class TimeDependent(NonTimeDependent):
    """
    A class for managing time-dependent solvers.

    This includes scenarios where the target is not directly a TimeFunction
    but depends on other functions that are.
    """
    @property
    def is_target_time(self):
        return True if any(i.is_Time for i in self.target.dimensions) else False

    def uxreplace_time(self, body, solver_objs):
        time_spacing = self.target.grid.stepping_dim.spacing
        true_dims = solver_objs['true_dims']

        time_mapper = {
            v: k.xreplace({time_spacing: 1, -time_spacing: -1})
            for k, v in solver_objs['time_mapper'].items()
        }
        subs = {symb: true_dims[time_mapper[symb]] for symb in time_mapper}
        return Uxreplace(subs).visit(body)

    def retrieve_time_dims(self, iters):
        time_iter = [i for i in iters if any(d.is_Time for d in i.dimensions)]
        mapper = {}
        if not time_iter:
            return mapper
        for d in time_iter[0].dimensions:
            if d.is_Modulo:
                mapper[d.origin] = d
            elif d.is_Time:
                mapper[d] = d
        return mapper

    def time_dep_replace(self, injectsolve, solver_objs, objs):
        # Extract the target from the lhs, which has been lowered
        # through the operator, allowing us to retrieve the target time symbol
        # from it
        target = self.injectsolve.expr.lhs

        if self.is_target_time:
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

            vec_replace_array = petsc_call(
                'VecReplaceArray', [solver_objs['x_local'], start_ptr]
            )
            return (vec_get_size, expr, vec_replace_array)
        else:
            field_from_ptr = FieldFromPointer(
                target.function._C_field_data, target.function._C_symbol
            )
            vec_replace_array = (petsc_call(
                'VecReplaceArray', [solver_objs['x_local'], field_from_ptr]
            ),)
            return vec_replace_array

    def assign_time_iters(self, iters, struct):
        """
        Assign time iterators to the struct.
        Ensure that assignment occurs only once per time loop, if necessary.
        Assign only the iterators that are common between the struct fields
        and the actual Iteration.
        """
        time_iter = [
            i for i in FindNodes(Iteration).visit(iters)
            if i.dim.is_Time
        ]
        assert len(time_iter) == 1
        time_iter, = time_iter

        common_dims = [d for d in time_iter.dimensions if d in struct.fields]
        common_dims = [
            DummyExpr(FieldFromComposite(d, struct), d) for d in common_dims
        ]
        return common_dims


Null = Macro('NULL')
void = 'void'


# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
