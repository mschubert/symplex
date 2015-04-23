import numpy as np
import sympy
import cplex

class CplexModel(cplex.Cplex):
    def __init__(self):
        cplex.Cplex.__init__(self)
        self._idx = {}

    def var(self, name, shape, lower_bound, upper_bound, vtype, obj):
        #FIXME: this should reference species name, not index
        ijstr = lambda i,j: sympy.Symbol(name+"_{"+str(int(i))+","+str(int(j))+"}")
        matrix = np.array(np.fromfunction(np.vectorize(ijstr), shape))

        names = [str(e) for e in matrix.flatten().tolist()]
        nvars = len(self._idx)
        ntup = zip(names, range(nvars, nvars + len(names)))
        self._idx.update({k:v for k,v in ntup})

        var = {}
        var['names'] = names
        var['ub'] = [upper_bound] * matrix.size
        var['lb'] = [lower_bound] * matrix.size
        var['types'] = [vtype] * matrix.size
        var['obj'] = [obj] * matrix.size
        self.variables.add(**var)

        return np.matrix(matrix)

    def constrain(self, expr, compare, rhs):
        cons = {}
        exprs = np.array(expr).flatten().tolist() #if e != 0]
        cons['lin_expr'] = [self._expr2list(e) for e in exprs]
        cons['senses'] = [compare] * len(exprs)

        if np.isscalar(rhs):
            cons['rhs'] = [rhs] * len(exprs)
        else:
            cons['rhs'] = np.array(rhs).flatten().tolist()

        self.linear_constraints.add(**cons)
#        return cons # debug

    def _expr2list(self, expr):
        var = list(expr.atoms(sympy.Symbol))
        coeff = [int(expr.coeff(v)) for v in var]
        var_index = [self._idx[str(v)] for v in var]
        return [var_index, coeff]

    def solve(self, threads=1, number_solutions=1, absgap=0.0,
            intensity=1, replace=0, mipgap=0, timelimit=3600):
        self.parameters.threads.set(threads)
        self.parameters.mip.limits.populate.set(int(number_solutions))
        self.parameters.mip.pool.absgap.set(absgap)
        self.parameters.mip.pool.intensity=intensity
        self.parameters.mip.pool.replace=replace
        self.parameters.mip.tolerances.mipgap.set(float(mipgap))
        self.parameters.timelimit.set(float(timelimit))

        # delegate call to parent class method
        cplex.Cplex.solve(self)

    def getSolution(self):
        pass
