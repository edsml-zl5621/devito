import pytest

from ctypes import c_void_p
import cgen
import sympy


from devito import (Eq, Grid, Function, TimeFunction, Operator, Dimension,  # noqa
                    switchconfig, dimensions, SpaceDimension)
from devito.ir.iet import (Call, Callable, Conditional, DummyExpr, Iteration, List,
                           Lambda, ElementalFunction, CGen, FindSymbols,
                           filter_iterations, make_efunc, retrieve_iteration_tree,
                           Definition, Expression, Transformer, CallBack)
from devito.ir import SymbolRegistry
from devito.passes.iet.engine import Graph, FindNodes
from devito.passes.iet.languages.C import CDataManager
from devito.symbolics import Byref, FieldFromComposite, InlineIf, Macro
from devito.tools import as_tuple
from devito.types import Array, LocalObject, Symbol
from devito.passes.iet.petsc import PetscObject
from devito.ir.equations import DummyEq


@pytest.fixture
def grid():
    return Grid((3, 3, 3))


@pytest.fixture
def fc(grid):
    return Array(name='fc', dimensions=(grid.dimensions[0], grid.dimensions[1]),
                 shape=(3, 5)).indexed


def test_conditional(fc, grid):
    x, y, _ = grid.dimensions
    then_body = DummyExpr(fc[x, y], fc[x, y] + 1)
    else_body = DummyExpr(fc[x, y], fc[x, y] + 2)
    conditional = Conditional(x < 3, then_body, else_body)
    assert str(conditional) == """\
if (x < 3)
{
  fc[x][y] = fc[x][y] + 1;
}
else
{
  fc[x][y] = fc[x][y] + 2;
}"""


@pytest.mark.parametrize("exprs,nfuncs,ntimeiters,nests", [
    (('Eq(v[t+1,x,y], v[t,x,y] + 1)',), (1,), (2,), ('xy',)),
    (('Eq(v[t,x,y], v[t,x-1,y] + 1)', 'Eq(v[t,x,y], v[t,x+1,y] + u[x,y])'),
     (1, 2), (1, 1), ('xy', 'xy'))
])
@switchconfig(openmp=False)
def test_make_efuncs(exprs, nfuncs, ntimeiters, nests):
    """Test construction of ElementalFunctions."""
    exprs = list(as_tuple(exprs))

    grid = Grid(shape=(10, 10))
    t = grid.stepping_dim  # noqa
    x, y = grid.dimensions  # noqa

    u = Function(name='u', grid=grid)  # noqa
    v = TimeFunction(name='v', grid=grid)  # noqa

    # List comprehension would need explicit locals/globals mappings to eval
    for i, e in enumerate(list(exprs)):
        exprs[i] = eval(e)

    op = Operator(exprs)

    # We create one ElementalFunction for each Iteration nest over space dimensions
    efuncs = []
    for n, tree in enumerate(retrieve_iteration_tree(op)):
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
        efuncs.append(make_efunc('f%d' % n, root))

    assert len(efuncs) == len(nfuncs) == len(ntimeiters) == len(nests)

    for efunc, nf, nt, nest in zip(efuncs, nfuncs, ntimeiters, nests):
        # Check the `efunc` parameters
        assert all(i in efunc.parameters for i in (x.symbolic_min, x.symbolic_max))
        assert all(i in efunc.parameters for i in (y.symbolic_min, y.symbolic_max))
        functions = FindSymbols().visit(efunc)
        assert len(functions) == nf
        assert all(i in efunc.parameters for i in functions)
        timeiters = [i for i in FindSymbols('basics').visit(efunc)
                     if isinstance(i, Dimension) and i.is_Time]
        assert len(timeiters) == nt
        assert all(i in efunc.parameters for i in timeiters)
        assert len(efunc.parameters) == 4 + len(functions) + len(timeiters)

        # Check the loop nest structure
        trees = retrieve_iteration_tree(efunc)
        assert len(trees) == 1
        tree = trees[0]
        assert all(i.dim.name == j for i, j in zip(tree, nest))

        assert efunc.make_call()


def test_nested_calls_cgen():
    call = Call('foo', [
        Call('bar', [])
    ])

    code = CGen().visit(call)

    assert str(code) == 'foo(bar());'


def test_callback_cgen():

    a = Symbol('a')
    b = Symbol('b')
    foo0 = Callable('foo0', Definition(a), 'void', parameters=[b])
    foo0_arg = CallBack(foo0.name, foo0.retval, 'int')
    code0 = CGen().visit(foo0_arg)
    assert str(code0) == '(void (*)(int))foo0'

    # test nested calls with CallBack (i.e a functionpointer) as the argument
    call = Call('foo1', [
        Call('foo2', [foo0_arg])
    ])
    code1 = CGen().visit(call)
    assert str(code1) == 'foo1(foo2((void (*)(int))foo0));'

    callees = FindNodes(Call).visit(call)
    assert len(callees) == 3


@pytest.mark.parametrize('mode,expected', [
    ('basics', '["x"]'),
    ('symbolics', '["f"]')
])
def test_find_symbols_nested(mode, expected):
    grid = Grid(shape=(4, 4, 4))
    call = Call('foo', [
        Call('bar', [
            Symbol(name='x'),
            Call('baz', [Function(name='f', grid=grid)])
        ])
    ])

    found = FindSymbols(mode).visit(call)

    assert [f.name for f in found] == eval(expected)


def test_list_denesting():
    l0 = List(header=cgen.Line('a'), body=List(header=cgen.Line('b')))
    l1 = l0._rebuild(body=List(header=cgen.Line('c')))
    assert len(l0.body) == 0
    assert len(l1.body) == 0
    assert str(l1) == "a\nb\nc"

    l2 = l1._rebuild(l1.body)
    assert len(l2.body) == 0
    assert str(l2) == str(l1)

    l3 = l2._rebuild(l2.body, **l2.args_frozen)
    assert len(l3.body) == 0
    assert str(l3) == str(l2)


def test_make_cpp_parfor():
    """
    Test construction of a CPP parallel for. This excites the IET construction
    machinery in several ways, in particular by using Lambda nodes (to generate
    C++ lambda functions) and nested Calls.
    """

    class STDVectorThreads(LocalObject):

        dtype = type('std::vector<std::thread>', (c_void_p,), {})

        def __init__(self):
            super().__init__('threads')

    class STDThread(LocalObject):

        dtype = type('std::thread&', (c_void_p,), {})

    class FunctionType(LocalObject):

        dtype = type('FuncType&&', (c_void_p,), {})

    # Basic symbols
    nthreads = Symbol(name='nthreads', is_const=True)
    threshold = Symbol(name='threshold', is_const=True)
    last = Symbol(name='last', is_const=True)
    first = Symbol(name='first', is_const=True)
    portion = Symbol(name='portion', is_const=True)

    # Composite symbols
    threads = STDVectorThreads()

    # Iteration helper symbols
    begin = Symbol(name='begin')
    l = Symbol(name='l')
    end = Symbol(name='end')

    # Functions
    stdmax = sympy.Function('std::max')

    # Construct the parallel-for body
    func = FunctionType('func')
    i = Dimension(name='i')
    threadobj = Call('std::thread', Lambda(
        Iteration(Call(func.name, i), i, (begin, end-1, 1)),
        ['=', Byref(func.name)],
    ))
    threadpush = Call(FieldFromComposite('push_back', threads), threadobj)
    it = Dimension(name='it')
    iteration = Iteration([
        DummyExpr(begin, it, init=True),
        DummyExpr(l, it + portion, init=True),
        DummyExpr(end, InlineIf(l > last, last, l), init=True),
        threadpush
    ], it, (first, last, portion))
    thread = STDThread('x')
    waitcall = Call('std::for_each', [
        Call(FieldFromComposite('begin', threads)),
        Call(FieldFromComposite('end', threads)),
        Lambda(Call(FieldFromComposite('join', thread.name)), [], [thread])
    ])
    body = [
        DummyExpr(threshold, 1, init=True),
        DummyExpr(portion, stdmax(threshold, (last - first) / nthreads), init=True),
        Call(FieldFromComposite('reserve', threads), nthreads),
        iteration,
        waitcall
    ]

    parfor = ElementalFunction('parallel_for', body,
                               parameters=[first, last, func, nthreads])

    assert str(parfor) == """\
static \
void parallel_for(const int first, const int last, FuncType&& func, const int nthreads)
{
  const int threshold = 1;
  const int portion = std::max(threshold, (-first + last)/nthreads);
  threads.reserve(nthreads);
  for (int it = first; it <= last; it += portion)
  {
    int begin = it;
    int l = it + portion;
    int end = (l > last) ? last : l;
    threads.push_back(std::thread([=, &func]()
    {
      for (int i = begin; i <= end - 1; i += 1)
      {
        func(i);
      }
    }));
  }
  std::for_each(threads.begin(),threads.end(),[](std::thread& x)
  {
    x.join();
  });
}"""


def test_make_cuda_stream():

    class CudaStream(LocalObject):

        dtype = type('cudaStream_t', (c_void_p,), {})

        @property
        def _C_init(self):
            return Call('cudaStreamCreate', Byref(self))

        @property
        def _C_free(self):
            return Call('cudaStreamDestroy', self)

    stream = CudaStream('stream')

    iet = Call('foo', stream)
    iet = ElementalFunction('foo', iet, parameters=())
    dm = CDataManager(sregistry=None)
    iet = CDataManager.place_definitions.__wrapped__(dm, iet)[0]

    assert str(iet) == """\
static void foo()
{
  cudaStream_t stream;
  cudaStreamCreate(&(stream));

  foo(stream);

  cudaStreamDestroy(stream);
}"""


def test_call_indexed():
    grid = Grid(shape=(10, 10))

    u = Function(name='u', grid=grid)

    foo = Callable('foo', DummyExpr(u, 1), 'void', parameters=[u, u.indexed])
    call = Call(foo.name, [u, u.indexed])

    assert str(call) == "foo(u_vec,u);"
    assert str(foo) == """\
void foo(struct dataobj *restrict u_vec, float *restrict u)
{
  u(x, y) = 1;
}"""


def test_call_retobj_indexed():
    grid = Grid(shape=(10, 10))

    u = Function(name='u', grid=grid)
    v = Function(name='v', grid=grid)

    call = Call('foo', [u], retobj=v.indexify())

    assert str(call) == "v[x][y] = foo(u_vec);"

    assert not call.defines


def test_null_init():
    grid = Grid(shape=(10, 10))

    u = Function(name='u', grid=grid)

    expr = DummyExpr(u.indexed, Macro('NULL'), init=True)

    assert str(expr) == "float * u = NULL;"
    assert expr.defines == (u.indexed,)


def test_templates():
    grid = Grid(shape=(10, 10))
    x, y = grid.dimensions

    u = Function(name='u', grid=grid)

    foo = Callable('foo', DummyExpr(u, 1), 'void', parameters=[u],
                   templates=[x, y])

    assert str(foo) == """\
template <int x, int y>
void foo(struct dataobj *restrict u_vec)
{
  u(x, y) = 1;
}"""


def test_codegen_quality0():
    grid = Grid(shape=(4, 4, 4))
    _, y, z = grid.dimensions

    a = Array(name='a', dimensions=grid.dimensions)

    expr = DummyExpr(a.indexed, 1)
    foo = Callable('foo', expr, 'void',
                   parameters=[a, y.symbolic_size, z.symbolic_size])

    # Emulate what the compiler would do
    graph = Graph(foo)

    CDataManager(sregistry=SymbolRegistry()).process(graph)

    foo1 = graph.root

    assert len(foo.parameters) == 3
    assert len(foo1.parameters) == 1
    assert foo1.parameters[0] is a


def test_petsc_object():

    obj1 = PetscObject(name='obj1', petsc_type='PetscInt')
    obj2 = PetscObject(name='obj2', petsc_type='PetscScalar')
    obj3 = PetscObject(name='obj3', petsc_type='Mat')
    obj4 = PetscObject(name='obj4', petsc_type='Vec')

    obj5 = PetscObject(name='obj5', petsc_type='PetscInt', is_const=True)
    obj6 = PetscObject(name='obj6', petsc_type='PetscScalar', grid=Grid((2,)))
    obj7 = PetscObject(name='obj7', petsc_type='PetscInt', grid=Grid((5,)))
    obj8 = PetscObject(name='obj8', petsc_type='PetscScalar', grid=Grid((5, 5)))
    obj9 = PetscObject(name='obj9', petsc_type='PetscInt', grid=Grid((5, 5)),
                       is_const=True)
    obj10 = PetscObject(name='obj10', petsc_type='PetscScalar', grid=Grid((10, 20, 30)))

    i, j, k = dimensions('i j k')
    obj11 = PetscObject(name='obj11', petsc_type='PetscInt', shape=(1, 1),
                        dimensions=(i, j))
    obj12 = PetscObject(name='obj12', petsc_type='PetscScalar', shape=(1, 1, 1),
                        dimensions=(i, j, k))

    defn1 = Definition(obj1)
    defn2 = Definition(obj2)
    defn3 = Definition(obj3)
    defn4 = Definition(obj4)
    defn5 = Definition(obj5)
    defn6 = Definition(obj6)
    defn7 = Definition(obj7)
    defn8 = Definition(obj8)
    defn9 = Definition(obj9)
    defn10 = Definition(obj10)
    defn11 = Definition(obj11)
    defn12 = Definition(obj12)

    assert str(defn1) == "PetscInt obj1;"
    assert str(defn2) == "PetscScalar obj2;"
    assert str(defn3) == "Mat obj3;"
    assert str(defn4) == "Vec obj4;"
    assert str(defn5) == "const PetscInt obj5;"
    assert str(defn6) == "PetscScalar * obj6;"
    assert str(defn7) == "PetscInt * obj7;"
    assert str(defn8) == "PetscScalar ** obj8;"
    assert str(defn9) == "const PetscInt ** obj9;"
    assert str(defn10) == "PetscScalar *** obj10;"
    assert str(defn11) == "PetscInt ** obj11;"
    assert str(defn12) == "PetscScalar *** obj12;"


def test_petsc_indexify():

    i, j = dimensions('i j')
    xarr = PetscObject(name='xarr', petsc_type='PetscScalar', shape=(50, 50),
                       dimensions=(i, j), is_const=True)

    tmp = PetscObject(name='tmp', petsc_type='PetscScalar')

    line1 = Definition(xarr)
    line2 = DummyExpr(tmp, xarr.indexify(indices=(25, 25)))
    line3 = DummyExpr(tmp, xarr.indexify())

    assert str(line1) == "const PetscScalar ** xarr;"
    assert str(line2) == "tmp = xarr[25][25];"
    assert str(line3) == "tmp = xarr[i][j];"


def test_petsc_expressions():

    a = PetscObject(name='a', petsc_type='PetscInt')
    expr1 = DummyExpr(a, 15, init=True)

    b = PetscObject(name='b', petsc_type='PetscScalar', is_const=True)
    c = PetscObject(name='c', petsc_type='PetscScalar', is_const=True)
    d = PetscObject(name='d', petsc_type='PetscScalar', is_const=True)
    expr2 = DummyExpr(b, c + d, init=True)
    expr3 = DummyExpr(b, c - d, init=True)
    expr4 = DummyExpr(b, c / d, init=True)
    expr5 = DummyExpr(b, c * d, init=True)
    expr6 = DummyExpr(b, c**-2, init=True)

    assert str(expr1) == "PetscInt a = 15;"
    assert str(expr2) == "const PetscScalar b = c + d;"
    assert str(expr3) == "const PetscScalar b = -d + c;"
    assert str(expr4) == "const PetscScalar b = c*1.0/d;"
    assert str(expr5) == "const PetscScalar b = c*d;"
    assert str(expr6) == "const PetscScalar b = pow(c, -2);"


def test_petsc_iterations():
    dims = {'x': Dimension(name='x'),
            'y': Dimension(name='y')}

    symbs = {'left': PetscObject(name='left', petsc_type='PetscScalar'),
             'right': PetscObject(name='right', petsc_type='PetscScalar'),
             'x_m': PetscObject(name='x_m', petsc_type='PetscInt', is_const=True),
             'x_M': PetscObject(name='x_M', petsc_type='PetscInt', is_const=True),
             'y_m': PetscObject(name='y_m', petsc_type='PetscInt', is_const=True),
             'y_M': PetscObject(name='y_M', petsc_type='PetscInt', is_const=True)}

    def get_exprs(left, right):
        return [Expression(DummyEq(left, 0.)),
                Expression(DummyEq(right, 0.))]

    exprs = get_exprs(symbs['left'],
                      symbs['right'])

    def get_iters(dims, symbs):
        return [lambda ex: Iteration(ex, dims['x'], (symbs['x_m'], symbs['x_M'], 1)),
                lambda ex: Iteration(ex, dims['y'], (symbs['y_m'], symbs['y_M'], 1))]

    iters = get_iters(dims, symbs)

    def get_block(exprs, iters):
        return iters[0](iters[1]([exprs[0], exprs[1]]))

    block1 = get_block(exprs, iters)

    kernel = Callable('foo', block1, 'void', ())

    assert str(kernel) == """\
void foo()
{
  for (int x = x_m; x <= x_M; x += 1)
  {
    for (int y = y_m; y <= y_M; y += 1)
    {
      left = 0.0;
      right = 0.0;
    }
  }
}"""


def test_petsc_dummy():
    """

    """
    # create an 'operator' manually
    dims_op = {'x': SpaceDimension(name='x'),
               'y': SpaceDimension(name='y')}

    grid = Grid(shape=(5, 5))
    symbs_op = {'u': Function(name='u', grid=grid).indexify()}

    def get_exprs(u):
        return [Expression(DummyEq(u, u+1))]

    exprs_op = get_exprs(symbs_op['u'])

    def get_iters(dims_op):
        return [lambda ex: Iteration(ex, dims_op['x'], (0, 4, 1)),
                lambda ex: Iteration(ex, dims_op['y'], (0, 4, 1))]

    iters_op = get_iters(dims_op)

    def get_block1(exprs_op, iters_op):
        return iters_op[0](iters_op[1](exprs_op[0]))

    block1 = get_block1(exprs_op, iters_op)

    kernel_op = Callable('kernel', block1, 'int', ())

    symbs_petsc = {'retval': PetscObject(name='retval', petsc_type='PetscErrorCode'),
                   'A_matfree': PetscObject(name='A_matfree', petsc_type='Mat'),
                   'xvec': PetscObject(name='xvec', petsc_type='Vec'),
                   'yvec': PetscObject(name='yvec', petsc_type='Vec'),
                   'xarr': PetscObject(name='xarr', petsc_type='PetscScalar',
                                       grid=Grid((2,)), is_const=True),
                   'yarr': PetscObject(name='yarr', petsc_type='PetscScalar',
                                       grid=Grid((2,)))}

    MyMatShellMult = Callable('MyMatShellMult', kernel_op.body,
                              retval=symbs_petsc['retval'],
                              parameters=(symbs_petsc['A_matfree'],
                                          symbs_petsc['xvec'], symbs_petsc['yvec']))

    call = Call(MyMatShellMult.name)
    transformer = Transformer({block1: call})
    main_block = transformer.visit(block1)
    new_op_block = [Call('PetscCall', [Call('VecGetArrayRead',
                                            arguments=[symbs_petsc['xvec'],
                                                       Byref(symbs_petsc['xarr'])])]),
                    main_block]
    main = Callable('main', new_op_block, 'int', ())

    assert('Original kernel:\n' + str(kernel_op) + '\n' +
           'MyMatShellMult with body of original kernel:\n' + str(MyMatShellMult) +
           '\n' + 'New kernel with a call to the MyMatShellMult function:\n' +
           str(main)) == """\
Original kernel:
int kernel()
{
  for (int x = 0; x <= 4; x += 1)
  {
    for (int y = 0; y <= 4; y += 1)
    {
      u[x][y] = u[x][y] + 1;
    }
  }
}
MyMatShellMult with body of original kernel:
PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec)
{
  for (int x = 0; x <= 4; x += 1)
  {
    for (int y = 0; y <= 4; y += 1)
    {
      u[x][y] = u[x][y] + 1;
    }
  }
}
New kernel with a call to the MyMatShellMult function:
int main()
{
  PetscCall(VecGetArrayRead(xvec,&xarr));
  MyMatShellMult();
}"""


def test_petsc_callable():

    retval = PetscObject(name='retval', petsc_type='PetscErrorCode')
    A_matfree = PetscObject(name='A_matfree', petsc_type='Mat')
    xvec = PetscObject(name='xvec', petsc_type='Vec')
    yvec = PetscObject(name='yvec', petsc_type='Vec')
    i, j, k = dimensions('i j k')
    xarr = PetscObject(name='xarr', petsc_type='PetscScalar', shape=(1,),
                       dimensions=(i,), is_const=True)
    yarr = PetscObject(name='yarr', petsc_type='PetscScalar', shape=(1,),
                       dimensions=(i,))
    nx = PetscObject(name='nx', petsc_type='PetscInt')
    tmp = PetscObject(name='tmp', petsc_type='PetscScalar')

    line1 = Definition(xarr)
    line2 = Definition(yarr)
    line3 = DummyExpr(nx, 15, init=True)
    line4 = Definition(tmp)

    # The VecGetArrayRead function provides a way to access the underlying
    # array of values stored in a 'Vec' object without making a copy of the data.
    line5 = Call('PetscCall', [Call('VecGetArrayRead', arguments=[xvec, Byref(xarr)])])
    line6 = Call('PetscCall', [Call('VecGetArray', arguments=[yvec, Byref(yarr)])])
    line7 = DummyExpr(tmp, xarr.indexify(indices=(50,)))
    line8 = DummyExpr(yarr.indexify(indices=(50,)), tmp)
    line9 = Call('PetscCall', [Call('VecRestoreArrayRead',
                                    arguments=[xvec, Byref(xarr)])])
    line10 = Call('PetscCall', [Call('VecRestoreArray', arguments=[yvec, Byref(yarr)])])
    line11 = Call('PetscFunctionReturn', 0)

    iet = [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11]

    MyMatShellMult = Callable('MyMatShellMult', iet, retval=retval,
                              parameters=(A_matfree, xvec, yvec))
    assert str(MyMatShellMult) == """\
PetscErrorCode MyMatShellMult(Mat A_matfree, Vec xvec, Vec yvec)
{
  const PetscScalar * xarr;
  PetscScalar * yarr;
  PetscInt nx = 15;
  PetscScalar tmp;
  PetscCall(VecGetArrayRead(xvec,&xarr));
  PetscCall(VecGetArray(yvec,&yarr));
  tmp = xarr[50];
  yarr[50] = tmp;
  PetscCall(VecRestoreArrayRead(xvec,&xarr));
  PetscCall(VecRestoreArray(yvec,&yarr));
  PetscFunctionReturn(0);
}"""
