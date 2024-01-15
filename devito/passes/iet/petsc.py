from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Expression, FindNodes, Section, List,
                           Callable, Call, Transformer, Callback,
                           Definition, Uxreplace, FindSymbols)
from devito.ir.equations.equation import OpAction
from devito.types.petsc import Mat, Vec, DM, PetscErrorCode, PETScStruct, PETScArray
from devito.symbolics import FieldFromPointer, Byref


__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):
    # NOTE: Currently only considering the case when opt='noop'. This
    # does not deal with the Temp Expressions generated when opt is not set
    # to 'noop' etc.

    # Find the Section containing the 'action'. For now, assume we're only solving
    # 1 equation via PETSc so only 1 action exists within a single Section.
    sections = FindNodes(Section).visit(iet)
    section_with_action = [
        sec
        for sec in sections
        if any(expr.operation is OpAction for expr in FindNodes(Expression).visit(sec))
    ][0]

    # Find the original target (i.e the field we are solving for)
    # TODO: Extend to multiple targets but for now assume
    # we are only solving 1 equation via PETSc.
    target = FindNodes(Expression).visit(section_with_action)
    target = [i for i in target[0].functions if not isinstance(i, PETScArray)][0]

    # Build PETSc objects required for the solve.
    petsc_objs = build_petsc_objects(target)

    # Build the struct that is needed within the matvec callback
    struct = build_struct(section_with_action)

    # Build the body of the matvec callback
    matvec_body = build_matvec_body(section_with_action, petsc_objs, struct)

    matvec_callback, solve_body = build_solve(matvec_body, petsc_objs, struct)

    # Replace target with a PETScArray inside the matvec callback function.
    mapper = {target.indexed: petsc_objs['xvec_tmp'].indexed}
    matvec_callback = Uxreplace(mapper).visit(matvec_callback)

    # Replace the Section that contains the action inside the Entry Function with
    # the corresponding PETSc calls.
    # TODO: Eventually, this will be extended to deal with multiple different
    # 'actions' associated with different equations to solve.
    iet = Transformer({section_with_action: solve_body}).visit(iet)

    return iet, {'efuncs': [matvec_callback]}


def build_petsc_objects(target):
    # TODO: Eventually, the objects built will be based
    # on the number of different PETSc equations present etc.

    return {'A_matfree': Mat(name='A_matfree'),
            'xvec': Vec(name='xvec'),
            'yvec': Vec(name='yvec'),
            'da': DM(name='da'),
            'x': Vec(name='x'),
            'err': PetscErrorCode(name='err'),
            'xvec_tmp': PETScArray(name='xvec_tmp', dtype=target.dtype,
                                   dimensions=target.dimensions,
                                   shape=target.shape, liveness='eager')}


def build_struct(action):
    # Build the struct
    tmp1 = FindSymbols('basics').visit(action)
    tmp2 = FindSymbols('dimensions|indexedbases').visit(action)
    usr_ctx = [symb for symb in tmp1 if symb not in tmp2]
    return PETScStruct('ctx', usr_ctx)


def build_matvec_body(action, objs, struct):
    get_context = Call('PetscCall', [Call('MatShellGetContext',
                                          arguments=[objs['A_matfree'],
                                                     Byref(struct.name)])])
    body = List(body=[Definition(struct),
                      get_context,
                      action])
    # Replace all symbols in the body that appear in the struct
    # with a pointer to the struct.
    for i in struct.usr_ctx:
        body = Uxreplace({i: FieldFromPointer(i, struct)}).visit(body)
    return body


def build_solve(matvec_body, petsc_objs, struct):

    matvec_callback = Callable('MyMatShellMult',
                               matvec_body,
                               retval=petsc_objs['err'],
                               parameters=(petsc_objs['A_matfree'],
                                           petsc_objs['xvec'],
                                           petsc_objs['yvec']))

    matvec_operation = Call('PetscCall', [Call('MatShellSetOperation',
                                               arguments=[petsc_objs['A_matfree'],
                                                          'MATOP_MULT',
                                                          Callback(matvec_callback.name,
                                                                   'void', 'void')])])

    set_context = Call('PetscCall', [Call('MatShellSetContext',
                                          arguments=[petsc_objs['A_matfree'],
                                                     struct])])

    body = List(body=[set_context,
                      matvec_operation])

    return matvec_callback, body
