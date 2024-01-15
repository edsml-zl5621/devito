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

    # Find the Section containing the 'action'.
    sections = FindNodes(Section).visit(iet)
    sections_with_action = []
    for section in sections:
        section_exprs = FindNodes(Expression).visit(section)
        if any(expr.operation is OpAction for expr in section_exprs):
            sections_with_action.append(section)

    # TODO: Extend to multiple targets but for now I am assuming we
    # are only solving 1 equation via PETSc.
    target = FindNodes(Expression).visit(sections_with_action[0])
    target = [i for i in target[0].functions if not isinstance(i, PETScArray)][0]

    # Build PETSc objects required for the solve.
    petsc_objs = build_petsc_objects(target)

    # Replace target with a PETScArray inside 'action'.
    mapper = {target.indexed: petsc_objs['xvec_tmp'].indexed}
    updated_action = Uxreplace(mapper).visit(sections_with_action[0])

    struct = build_struct(updated_action)

    # Build the body of the matvec callback
    matvec_body = build_matvec_body(updated_action, petsc_objs, struct)

    matvec_callback, solve_body = build_solve(matvec_body, petsc_objs, struct)

    # TODO: Eventually, this will be extended to deal with multiple
    # 'actions' associated with different equations to solve.
    iet = Transformer({sections_with_action[0]: solve_body}).visit(iet)

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
    usr_ctx = [i for i in tmp1 if i not in tmp2]

    return PETScStruct('ctx', usr_ctx)


def build_matvec_body(action, objs, struct):

    get_context = Call('PetscCall', [Call('MatShellGetContext',
                                          arguments=[objs['A_matfree'],
                                                     Byref(struct.name)])])
    body = List(body=[Definition(struct),
                      get_context,
                      action])
    # Replace all symbols in the body that appear in the struct
    # with a pointer to the struct
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
