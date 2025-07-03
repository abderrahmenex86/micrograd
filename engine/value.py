class Value:

    def __init__(self, value,  children=(), op="", label=''):
        self.value = value
        self.label = label
        self.children = set(children)
        self.op = op

    def __repr__(self):
        return f"Value(value={self.value})"

    def __add__(self, other):
        if isinstance(other, Value):
            out = Value(self.value + other.value, (self, other), "+")
            return out
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Value):
            out = Value(self.value * other.value, (self, other), "*")
            return out
        return NotImplemented
