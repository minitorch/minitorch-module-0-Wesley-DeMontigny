"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y

def id(x):
    return x

def add(x: float, y:float) -> float:
    return x + y

def neg(x: float) -> float:
    return -1.0 * x

def lt(x: float, y:float) -> bool:
    return x < y

def eq(x:float, y:float) -> bool:
    return x == y

def max(x:float, y:float) -> float:
    return y if lt(x, y) else x

def is_close(x:float, y:float) -> bool:
    return abs(x - y) < 1e-2

def sigmoid(x:float) -> float:
    if x >= 0:
        return 1/(1 + math.exp(-1.0*x))
    else:
        exp_x = math.exp(x)
        return (exp_x)/(1 + exp_x)

def relu(x: float) -> float:
    return x if x > 0 else 0

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    return 1/x

def log_back(x: float, y: float) -> float:
    return y/x

def inv_back(x: float, y: float) -> float:
    return y/(x**2)

def relu_back(x: float, y:float) -> float:
    return 0 if x <= 0 else y

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def map_ret(a: Iterable[float]) -> Iterable[float]:
        ret = []
        for i in a:
            ret.append(fn(i))
        return ret
    return map_ret

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def zip_ret(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
        ret = []
        for i in range(len(a)):
            ret.append(fn(a[i], b[i]))
        return ret
    return zip_ret

def reduce(fn: Callable[[Iterable[float]], float], start: float) -> Callable[[Iterable[float]], float]:
    def reduce_ret(a: Iterable[float]) -> float:
        ret = start
        for i in a:
            ret = fn(ret, i)
        return ret
    return reduce_ret

def negList(a: Iterable[float]) -> Iterable[float]:
    return map(neg)(a)

def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(a, b)

def sum(a: Iterable[float]) -> float:
    return reduce(add, 0)(a)

def prod(a: Iterable[float]) -> float:
    return reduce(mul, 1)(a)
