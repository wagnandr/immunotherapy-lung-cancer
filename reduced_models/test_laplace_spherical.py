import dolfin as df
from coordinate_systems import SphericalCoordinateSystem

N = 100
r_max = 2

mesh = df.IntervalMesh(N, 0, r_max)

V = df.FunctionSpace(mesh, 'P', 2)

r = df.SpatialCoordinate(mesh)

u = df.TrialFunction(V)
v = df.TestFunction(V)

coordinates = SphericalCoordinateSystem(mesh)

spherical_laplace = coordinates.stiffness_form
spherical_mass = coordinates.mass_form

#m = lambda u: df.Constant(1)
m = df.exp(r**2)
a = spherical_laplace(m, u, v)
#f_fun = df.Constant('-6')
f_fun = df.Expression('- (4 * x[0]*x[0] + 6) * exp( x[0]*x[0] )', degree=8)
f = spherical_mass(f_fun, v)

bv = df.Expression('x[0]*x[0]', degree=2)
bi = df.CompiledSubDomain('on_boundary')
bc = df.DirichletBC(V, bv, bi)

u = df.Function(V)

df.solve(a == f, u, [bc])

print( df.errornorm(df.Expression('x[0]*x[0]', degree=2), u) )

from matplotlib import pyplot as plt
df.plot(u)
plt.show()
