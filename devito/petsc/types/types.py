import sympy

from devito.tools import Reconstructable, sympy_mutex, as_tuple


class LinearSolver(sympy.Function, Reconstructable):

    __rargs__ = ('expr',)
    __rkwargs__ = ('solver_parameters', 'fielddata', 'parent_dm', 'children_dms',
                   'time_mapper')

    defaults = {
        'ksp_type': 'gmres',
        'pc_type': 'jacobi',
        'ksp_rtol': 1e-7,  # Relative tolerance
        'ksp_atol': 1e-50,  # Absolute tolerance
        'ksp_divtol': 1e4,  # Divergence tolerance
        'ksp_max_it': 10000  # Maximum iterations
    }

    def __new__(cls, expr, solver_parameters=None,
                fielddata=None, parent_dm=None, children_dms=None,
                time_mapper=None, **kwargs):

        if solver_parameters is None:
            solver_parameters = cls.defaults
        else:
            for key, val in cls.defaults.items():
                solver_parameters[key] = solver_parameters.get(key, val)

        with sympy_mutex:
            obj = sympy.Function.__new__(cls, expr)

        obj._expr = expr
        obj._solver_parameters = solver_parameters
        obj._fielddata = fielddata if fielddata else FieldData()
        obj._parent_dm = parent_dm
        obj._children_dms = children_dms
        obj._time_mapper = time_mapper
        return obj

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.expr)

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    def __hash__(self):
        return hash(self.expr)

    def __eq__(self, other):
        return (isinstance(other, LinearSolver) and
                self.expr == other.expr)

    @property
    def expr(self):
        return self._expr

    @property
    def solver_parameters(self):
        return self._solver_parameters

    @property
    def fielddata(self):
        return self._fielddata

    @property
    def parent_dm(self):
        """DM attached to the SNES object and responsible for the solve"""
        return self._parent_dm

    @property
    def children_dms(self):
        """DMs associated with each field in the solve"""
        children_dms = self._children_dms
        return children_dms if children_dms is not None else as_tuple(self.parent_dm)

    @property
    def dms(self):
        dmdas = [self.parent_dm]
        # Add children_dms if the parent_dm is not in them
        if self.parent_dm not in self.children_dms:
            dmdas.extend(self.children_dms)
        return list(set(dmdas))

    @property
    def time_mapper(self):
        return self._time_mapper

    @classmethod
    def eval(cls, *args):
        return None

    func = Reconstructable._rebuild


# TODO: Reconstructable?
class FieldData:
    def __init__(self, target=None, matvecs=None, formfuncs=None, formrhs=None,
                 arrays=None, dmda=None, **kwargs):

        self.target = target
        self.matvecs = matvecs
        self.formfuncs = formfuncs
        self.formrhs = formrhs
        self.arrays = arrays
        self.dmda = dmda


class FieldDataNest(FieldData):
    def __init__(self):
        self.field_data_list = []

    def add_field_data(self, field_data):
        self.field_data_list.append(field_data)

    def get_field_data(self, target):
        for field_data in self.field_data_list:
            if field_data.target == target:
                return field_data
        raise ValueError("FieldData with target %s not found." % target)
    pass

    @property
    def targets(self):
        return tuple(field_data.target for field_data in self.field_data_list)
