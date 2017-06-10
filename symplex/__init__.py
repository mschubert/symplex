import numpy as np
import symengine as sympy
import cplex

class CplexModel(cplex.Cplex):
    def __init__(self):
        """Initializes the model

        In addition, creates empty class-internal dictionaries to look up variable
        names to their index in the CPlex model (`self._idx`) and shape (`self._shape`).
        """
        cplex.Cplex.__init__(self)
        self._idx = {} # name index
        self._shape = {} # variable index

    def var(self, name, shape, lower_bound, upper_bound, vtype, obj):
        """Add a variable to the model

        This method will create a matrix with dimensions `shape` that is filled
        with SymPy symbols of name `name_{i,j}`, where `i` and `j` are indices
        along the rows and columns, respectively. This is the return value.

        It will also create convert CPlex-structured variables for the names,
        upper- and lower bounds, types, and **obj? and add them to the CPlex model
        it is derived from.

        name -- The name of the variable; will be indexed as "name_{i,j}"
        shape -- A tuple with the dimension lengths [rows x columns]
        lower_bound -- The lower value limit a variable can take
        upper_bound -- The upper value limit a variable can take
        vtype -- Type of the variable; 'I' integer, 'B' binary; 'C' count (I>=0)
        obj --
        return -- A numpy matrix with shape `shape` filled with SymPy symbols `name_{i,j}`
        """
        if name in self._shape.keys():
            raise("Variable with name {} already added".format(name))
        else:
            self._shape[name] = shape

        #TODO: this should reference species name, not index
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

        if np.isscalar(obj):
            var['obj'] = [obj] * matrix.size
        else:
            var['obj'] = np.array(obj).flatten().tolist()
        #TODO: check if non-scalar args have the right length

        self.variables.add(**var)
        return np.matrix(matrix)

    def constrain(self, expr, compare, rhs):
        """ Method to add linear constraints to the model

        Consider the equation `x + y < 3`. `expr` is `x + y`, `compare` is the
        comparison operator `<`, and `3` is the `rhs`.

        If `rhs` is scalar but the expression is vector-valued, it will be
        interpreted that every value of the vector has to be true for `rhs`.

        expr -- A mathematical expression containing coefficients and variables
                created using the `var()` method
        compare -- 'G', 'L', 'E', for greater, less than, or equal, respectively
        rhs -- The integer value of the right-hand side of the equation
        """
        if compare not in ['G', 'L', 'E']:
            raise("Invalid comparison operator: only 'G', 'L', and 'E'")

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
        """Helper function to extract variables and coefficients from an expression

        expr -- An expression defined by coefficients and SymPy variables
        return -- A list consisting of two elements: the index a variable has in
                  the model, and its coefficient in the expression
        """
        coeff = []
        var_index = []
        for arg in expr.args: #FIXME: this is still slow, but not as bad
            if isinstance(arg, sympy.Symbol):
                coeff.append(1)
                var_index.append(self._idx[str(arg)])
            else:
                coeff.append(int(arg.args[0]))
                var_index.append(self._idx[str(arg.args[1])])

#        var = list(expr.atoms(sympy.Symbol))
#        coeff = [int(expr.coeff(v)) for v in var] #FIXME: this is too slow
#        var_index = [self._idx[str(v)] for v in var]
        return [var_index, coeff]

    def solve(self, threads=1, number_solutions=1, absgap=0.0,
            intensity=1, replace=0, mipgap=0, timelimit=3600):
        self.parameters.threads.set(threads)
        self.parameters.mip.limits.populate.set(int(number_solutions))
        self.parameters.mip.pool.absgap.set(absgap)
        self.parameters.mip.pool.intensity = intensity
        self.parameters.mip.pool.replace = replace
        self.parameters.mip.tolerances.mipgap.set(float(mipgap))
        self.parameters.timelimit.set(float(timelimit))

        # delegate call to parent class method
        cplex.Cplex.solve(self)
        self.populate_solution_pool()

    def get_solution(self, num=0):
        sol = self.solution.pool.get_values(num)
        solDict = {}
        for name,shape in self._shape.items():
            solDict[name] = np.empty(shape, dtype='int8')
            ijstr = lambda i,j: name+"_{"+str(int(i))+","+str(int(j))+"}"
            for i in range(shape[0]):
                for j in range(shape[1]):
                    solDict[name][i,j] = sol[self._idx[ijstr(i,j)]]
        return solDict
