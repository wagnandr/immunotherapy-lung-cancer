import numpy as np
import dolfin as df


def IC_norm(x,b):
    return sum((x[i] - b[i])**2 for i in range(len(x)))


def IC_smooth(x,a,b): 
    if IC_norm(x,b) < a**2: 
        return np.exp(1 - 1/(1 - IC_norm(x,b)/a**2))
    else: return 0


class InitialConditionCircle(df.UserExpression):
    def __init__(self, radius, midpoint, *args, **kwargs):
        self.radius = radius
        self.midpoint = midpoint 
        super().__init__(*args, **kwargs)

    def eval(self, values, x):
        values[0] = IC_smooth(x, self.radius, self.midpoint)
        values[1] = 0.0 
    def value_shape(self):
        return (2,)


def ExpressionSmoothCircle(midpoint: df.Point, r: float, a: float, degree: int):
    r_str = f'std::sqrt(std::pow(x[0]-({midpoint[0]}),2) + std::pow(x[1]-({midpoint[1]}),2) + std::pow(x[2]-({midpoint[2]}),2))'
    return df.Expression((f'1. / (exp( a * ({r_str}-r) ) + 1)', '0'), r=r, a=a, degree=degree)
    

def refine_locally(mesh, midpoint, refinement_radius, num_refinements):
    tumor_domain = df.CompiledSubDomain(f'(x[0]-({midpoint[0]}))*(x[0]-({midpoint[0]})) + (x[1]-({midpoint[1]}))*(x[1]-({midpoint[1]})) + (x[2]-({midpoint[2]}))*(x[2]-({midpoint[2]}))<= {refinement_radius}*{refinement_radius}')
    for i in range(num_refinements):
        marker = df.MeshFunction('bool', mesh, mesh.topology().dim())
        tumor_domain.mark(marker, True)
        mesh = df.refine(mesh, marker)
    return mesh