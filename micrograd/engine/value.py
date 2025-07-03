import math


class Value:

    def __init__(self, value, children=(), op="", label=""):
        self.value = value
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label
        self.children = set(children)
        self.op = op

    def __repr__(self):
        return f"Value(value={self.value})"

    def __add__(self, other):
        assert isinstance(
            other, (Value, int, float)
        ), "Can only add Value or numeric types"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        assert isinstance(other, (int, float)), "Can only add numeric types"
        return self + Value(other)

    def __neg__(self):
        return self * Value(-1)

    def __sub__(self, other):
        assert isinstance(
            other, (Value, int, float)
        ), "Can only subtract Value or numeric types"
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other):
        assert isinstance(other, (int, float)), "Can only subtract numeric types"
        return other + (-self)

    def __mul__(self, other):
        assert isinstance(other, (Value)), "Can only multiply Value or numeric types"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other), "*")

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        assert isinstance(other, (int, float)), "Can only multiply numeric types"
        return self * Value(other)

    def __truediv__(self, other):
        assert isinstance(other, (Value)), "Can only divide by Value or numeric types"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value / other.value, (self, other), "/")

        def _backward():
            self.grad += (1 / other.value) * out.grad
            other.grad -= (self.value / (other.value**2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        assert isinstance(other, (int, float)), "Can only divide numeric types"
        return Value(other) / self

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only supports int/float powers"
        out = Value(self.value**exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += (exponent * self.value ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.value
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.value * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.value
        t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topological = []
        visited = set()

        def build_topological_order(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topological_order(child)
                topological.append(node)

        build_topological_order(self)

        self.grad = 1.0
        for node in reversed(topological):
            node._backward()
