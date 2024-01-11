from devito.passes.iet.engine import iet_pass
from devito.ir.iet import Expression, FindNodes, Section
from devito.ir.equations.equation import OpAction


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

    return iet, {}
