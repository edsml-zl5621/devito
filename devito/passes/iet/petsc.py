from devito.passes.iet.engine import iet_pass

__all__ = ['lower_petsc']


@iet_pass
def lower_petsc(iet, **kwargs):

    # TODO: Drop the LinearSolveExpr's using .args[0] so that _rebuild doesn't
    # appear in ccode

    # Check if PETScSolve was used and count occurrences. Each PETScSolve
    # will have a unique MatVecAction.
    is_petsc = FindNodes(MatVecAction).visit(iet)

    if is_petsc:

        # Collect all solution fields we're solving for
        targets = [i.expr.rhs.target for i in is_petsc]

        # Initialize PETSc i.e generate the one off PETSc calls.
        init_setup = petsc_setup(targets, **kwargs)

        # TODO: Insert code that utilises the metadata attached to each LinSolveExpr
        # that appears in the RHS of each LinearSolverExpression.

        # Remove the LinSolveExpr that was utilised above to carry metadata.
        mapper = {expr:
                  expr._rebuild(expr=expr.expr._rebuild(rhs=expr.expr.rhs.expr))
                  for expr in FindNodes(LinearSolverExpression).visit(iet)}

        iet = Transformer(mapper).visit(iet)

        body = iet.body._rebuild(body=(tuple(init_setup) + iet.body.body))
        iet = iet._rebuild(body=body)

    return iet, {}


