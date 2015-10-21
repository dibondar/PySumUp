"""
Sum up taylor series by using genetic programming.

This little script is devoted to Andre Campos, for his awesome ability to sum up the series.
Even though this script cannot automatize Andre's talent, it might be still of some help.
"""

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import sympy
from sympy.core.basic import Basic
from sympy.core.numbers import NaN
from sympy.core.numbers import Infinity
from sympy.core.numbers import ComplexInfinity

import itertools
import operator
import string
import random

import numpy


class PySumUp:
    """
    Perform summation of taylor expansion via genetic programming
    """
    def __init__(self, **kwargs):
        """
        :param series_to_sum: (mandatory) sympy expression to be summed up
        :param var: sympy symbol with respect, which series is obtained
        :param x0: at which point to taylor expand
        :param order: number of terms in expansion
        :param min_expr_depth: (optional) Min complexity of randomly generated expression
        :param max_expr_depth: (optional) Max complexity of randomly generated expression
        :param MU: (optional) population size
        :param NGEN: (optional) number of generation for genetic algorithm (GA)
        :param LAMBDA: (optional) number of offspring generated at each generation of GA
        :param CXPB: (optional) probability of crossover in GA
        :param MUTPB: (optional) mutation probability
        """

        # save all the arguments
        self.kwargs = kwargs

        # save the arguments
        self.var = self.kwargs["var"]
        self.x0 = self.kwargs["x0"]
        self.order = self.kwargs["order"]
        self.series_to_sum = self.kwargs["series_to_sum"]

        # partition numpy expression
        self.partitioned_series = tuple(self.series2list(self.series_to_sum))

        # Set up genetic programming
        self.set_up_gp()

    def series2list(self, expr):
        """
        Convert a sympy expression into list, where each term corresponds to different order in taylor series
        :param expr: sympy expression to partition
        """
        partition = []
        for n in xrange(self.order):
            partition.append(
                expr.subs(self.var, self.x0).simplify()
            )
            expr = expr.diff(self.var, 1).simplify()
        #
        return partition

    @staticmethod
    def symbolic_norm(C):
        """
        Get "norm" of sympy expression to characterize complexity of complexity of C
        """
        return sum(not(S.is_Atom) for S in sympy.postorder_traversal(C))

    @staticmethod
    def evalf_norm(C):
        """
        Get "norm" of sympy expression to characterize complexity of complexity of C
        """
        x = numpy.float(abs(C.evalf()))
        x = (x if not numpy.isnan(x) else numpy.inf)
        return x

    def match_series(self, expr):
        """
        Generator to get fitness for deap programing
        :param expr: sympy expression to be benchmarked with respect to self.partitioned_series
        """

        # counter of mismatched terms
        num_mismatch_terms = 0

        # value of mismatching
        mismatch = 0

        for s_term in self.partitioned_series:
            if num_mismatch_terms < 3 and numpy.isfinite(mismatch):
                # if no mismatch found, continue evaluation
                try:
                    # evaluate at self.x0
                    subs_expr = expr.subs(self.var, self.x0).simplify()

                    if isinstance(subs_expr, (NaN, Infinity, ComplexInfinity)):
                        # invalid expression obtained, return infinite mismatch
                        mismatch = numpy.inf
                    else:
                        delta = subs_expr - s_term
                        # quantify term mismatching
                        # if terms matched perfectly 0, else get the norm of difference
                        mismatch = (0 if delta == 0 else self.norm(delta) + 1)

                        if mismatch or num_mismatch_terms:
                            # mismatching term is found
                            num_mismatch_terms += 1

                        # differentiate for the next step
                        expr = expr.diff(self.var, 1)
                except Exception:
                    # there were some errors in evaluation
                    mismatch = numpy.inf
            yield mismatch

    def eval_fitness(self, individual):
        """
        Evaluate fitness function for deap genetic programming
        """
        # transform the tree expression in a callable function
        try:
            expr = self.toolbox.compile(expr=individual)(self.var)
        except(ZeroDivisionError, OverflowError):
            expr = None

        if not(isinstance(expr, Basic) and self.var in expr.free_symbols):
            # expr is not a sympy object
            match = tuple(numpy.inf for _ in xrange(self.order))
        else:
            match = tuple(self.match_series(expr))
            # for debugging:
            # print expr, match

        if sum(match) == 0:
            # perfect matching is found
            self.exact_sum = expr
            print("Exact sum is found!!! %s\n"% str(self.exact_sum))

        return match

    def set_primitive_set(self):
        """
        Set up deap set of primitives, needed for genetic programming
        """
        self.pset = gp.PrimitiveSet(self.get_unique_str(), 1)
        self.pset.renameArguments(ARG0=str(self.var))

        # add these symbols into primitive set
        for S in self.series_to_sum.free_symbols - {self.var}:
            self.pset.addTerminal(S, name=str(S))

        # Add basic operations into the
        self.use_func(
            (operator.mul, 2), (operator.div, 2), (operator.sub, 2), (operator.add, 2)
        )

        # find unique number from series
        unique_num = set()
        for s_term in self.partitioned_series:
            unique_num.update(S for S in sympy.postorder_traversal(s_term) if S.is_Number)

        # convert numbers into fractions and extract nominator and denominator separately
        unique_num = itertools.chain(*(sympy.fraction(S) for S in unique_num))
        self.unique_num = sorted(set(unique_num))

        return self

    def use_elementary_func(self):
        """
        Add elementary functions (exp, cos, ...) to set of primitives for genetic programming
        """
        return self.use_unary_func(
            sympy.exp, sympy.sqrt, sympy.log,
            sympy.sinh, sympy.sin, sympy.cosh, sympy.cos,
            sympy.tanh, sympy.tan, sympy.cot, sympy.coth,
            sympy.acos, sympy.asin, sympy.atan, sympy.acot,
            sympy.acosh, sympy.asinh, sympy.atanh, sympy.acoth
        )

    def use_unary_func(self, *args):
        return self.use_func(*zip(args, itertools.repeat(1, len(args))))

    def use_func(self, *args):
        """
        Add the operations to set of primitives for genetic programming
        :param args: list of tuples (function, number of arguments)
        """
        for f, n in args:
            self.pset.addPrimitive(f, n)
        return self

    def use_consts(self, *args):
        """
        Add constants to set of primitives for genetic programming
        """
        for c in args:
            self.pset.addTerminal(c, name=str(c))
        return self

    @staticmethod
    def get_unique_str():
        """
        Return unique string
        """
        return ''.join(random.choice(string.ascii_letters) for _ in xrange(30))

    def use_rand_int(self, min_=1, max_=10):
        """
        Add ephemeral integer
        """
        # We need to generate unique name for ephemeral constant as required by DEAP
        is_error = True
        while is_error:
            try:
                self.pset.addEphemeralConstant(self.get_unique_str(), lambda: random.randint(min_, max_))
                is_error = False
            except Exception:
                pass
        return self

    def set_up_gp(self):
        """
        Set up deap genetic programming
        """
        # get set of primitives
        self.set_primitive_set()

        # Extract values
        min_expr_depth = self.kwargs.get("min_expr_depth", 1)
        max_expr_depth = self.kwargs.get("max_expr_depth", 2)

        creator.create("FitnessMin", base.Fitness, weights=tuple(-1 for _ in xrange(self.order)))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genFull, pset=self.pset, min_=min_expr_depth, max_=max_expr_depth)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        # select the fitness function
        self.toolbox.register("evaluate", self.eval_fitness)

        self.toolbox.register("select", tools.selBest)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=min_expr_depth, max_=max_expr_depth)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        # Set limits of expressions
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        # Select "norm" of expression, needed for the fitness function
        if self.series_to_sum.free_symbols == {self.var}:
            # since there are no, free symbols, just numbers, use float norm
            self.norm = self.evalf_norm
        else:
            # otherwise use symbolic norm
            self.norm = self.symbolic_norm

        # Set statistics
        self.stats = tools.Statistics(lambda ind: numpy.count_nonzero(ind.fitness.values))

        self.stats.register("min number of unmatched terms", numpy.min)
        self.stats.register("max number of unmatched terms", numpy.max)

        self.stats.register("std", numpy.std)
        self.stats.register("avg", numpy.mean)

        # Set hall of fame
        self.hof = tools.HallOfFame(10)

        # Initialize population
        self.MU = self.kwargs.get("MU", 100)
        self.pop = self.toolbox.population(n=self.MU)

        return self

    def run(self):
        """
        Start genetic programming to summ up t
        """
        NGEN = self.kwargs.get("NGEN", 10)
        LAMBDA = self.kwargs.get("LAMBDA", 100)
        CXPB = self.kwargs.get("CXPB", 0.5)
        MUTPB = self.kwargs.get("MUTPB", 0.3)

        algorithms.eaMuPlusLambda(
            self.pop, self.toolbox, self.MU, LAMBDA, CXPB, MUTPB, NGEN, self.stats, halloffame=self.hof
        )
        return self

    def print_hof(self):
        """
        print current hall of fame
        """
        print("================ Current hall of fame ==============")

        for ind in self.hof:
            print self.toolbox.compile(expr=ind)(self.var)

        print("====================================================")

        return self

    def print_pop(self):
        """
        print current hall of fame
        """
        print("================ Current population ==============")

        for ind in self.pop:
            print self.toolbox.compile(expr=ind)(self.var)

        print("====================================================")

        return self

if __name__ == '__main__':

    # Test
    x, s = sympy.symbols('x s')

    S = sympy.exp(sympy.sqrt(x + 2))
    n = 10
    """
    test1 = PySumUp(series_to_sum=S.series(x, 0, n), var=x, x0=0, order=n)\
                .use_unary_func(sympy.exp, sympy.sqrt).run().print_hof()
    """
    test2 = PySumUp(series_to_sum=S.series(x, 0, n), var=x, x0=0, order=n)\
            .use_consts(1, 2).use_unary_func(sympy.sqrt, sympy.exp)\
            .run().print_hof()