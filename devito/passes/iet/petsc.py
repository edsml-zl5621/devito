from devito.passes.iet.engine import iet_pass
from devito.ir.iet import Expression, FindNodes, Section, List, Callable, Call, Transformer, Callback
from devito.ir.equations.equation import OpAction
from devito.types.petsc import Mat, Vec, DM, PetscErrorCode


__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # Find the section containing the 'action'
    sections = FindNodes(Section).visit(iet)
    sections_with_action = []
    for section in sections:
        section_exprs = FindNodes(Expression).visit(section)
        if any(expr.operation is OpAction for expr in section_exprs):
            sections_with_action.append(section)

    petsc_objs = {'A_matfree': Mat(name='A_matfree'),
                  'xvec': Vec(name='xvec'),
                  'yvec': Vec(name='yvec'),
                  'da': DM(name='da'),
                  'x': Vec(name='x'),
                  'err': PetscErrorCode(name='err')}
    
    mat_vec_callback = Callable('MyMatShellMult', sections_with_action, retval=petsc_objs['err'],
                                parameters=(petsc_objs['A_matfree'], petsc_objs['xvec'], petsc_objs['yvec']))
    
    matvec_operation = Call('PetscCall', [Call('MatShellSetOperation',
                                                   arguments=[petsc_objs['A_matfree'],
                                                              'MATOP_MULT',
                                                              Callback(mat_vec_callback.name, 'void', 'void')])])

    iet = Transformer({sections_with_action[0]: List(body=[matvec_operation])}).visit(iet)

    return iet, {'efuncs': [mat_vec_callback]}

