import dolfin as df
from coordinate_systems import CylindricalCoordinateSystem

N = 100
r_max = 2
z_max = 2

mesh = df.RectangleMesh(df.Point(0, -z_max), df.Point(r_max, z_max), N, N)

V = df.FunctionSpace(mesh, 'P', 2)

r, z = df.SpatialCoordinate(mesh)

u = df.TrialFunction(V)
v = df.TestFunction(V)

coordinates = CylindricalCoordinateSystem(mesh)

a = coordinates.stiffness_form
m = coordinates.mass_form

mob = df.exp(r**2 + z)
#mob = df.Constant(1)
L = a(mob, u, v) 
#f_fun = df.Constant('-6')
f_fun = df.Expression('- (4 * x[0]*x[0] + 6 * x[1] + 10) * exp( x[0]*x[0] + x[1] )', degree=8)
f = m(f_fun, v)

bv = df.Expression('x[0]*x[0] + 3*x[1]*x[1]', degree=2)
bi = df.CompiledSubDomain('on_boundary')
bc = df.DirichletBC(V, bv, bi)

u = df.Function(V, name='u')

df.solve(L == f, u, [bc])

print( df.errornorm(df.Expression('x[0]*x[0] + 3*x[1]*x[1]', degree=2), u) )

df.File('u.pvd') << u