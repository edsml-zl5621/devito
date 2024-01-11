from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Expression, FindNodes, Section, List,
                           Callable, Call, Transformer, Callback)
from devito.ir.equations.equation import OpAction
from devito.types.petsc import Mat, Vec, DM, PetscErrorCode


__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # Find the Section containing the 'action'
    sections = FindNodes(Section).visit(iet)
    secs_with_action = []
    for section in sections:
        section_exprs = FindNodes(Expression).visit(section)
        if any(expr.operation is OpAction for expr in section_exprs):
            secs_with_action.append(section)

    petsc_objs = build_petsc_objects()

    matvec_callback = Callable('MyMatShellMult', secs_with_action[0],
                               retval=petsc_objs['err'],
                               parameters=(petsc_objs['A_matfree'],
                                           petsc_objs['xvec'],
                                           petsc_objs['yvec']))

    matvec_operation = Call('PetscCall', [Call('MatShellSetOperation',
                                               arguments=[petsc_objs['A_matfree'],
                                                          'MATOP_MULT',
                                                          Callback(matvec_callback.name,
                                                                   'void', 'void')])])

    # TODO: Eventually, this will be extended to deal with multiple
    # 'actions' associated with different equations to solve.
    iet = Transformer({secs_with_action[0]: List(body=[matvec_operation])}).visit(iet)

    return iet, {'efuncs': [matvec_callback]}


def build_petsc_objects():
    # TODO: Eventually, the objects built will be based
    # on the number of different PETSc equations present etc.

    return {'A_matfree': Mat(name='A_matfree'),
            'xvec': Vec(name='xvec'),
            'yvec': Vec(name='yvec'),
            'da': DM(name='da'),
            'x': Vec(name='x'),
            'err': PetscErrorCode(name='err')}
