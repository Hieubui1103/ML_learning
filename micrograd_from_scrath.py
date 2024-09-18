import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5

print(f(3.0))

xs = np.arange(-5,5,0.25)
print(xs)
ys = f(xs)
print(ys)
plt.plot(xs,ys)

plt.savefig("sample_plot.png")

h = 0.0000000001
x = 2/3
print((f(x+h)-f(x))/h)

# Let's get more complex
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)

#--------------------
h = 0.0001
# inputs
a = 2.0
b = -3.0
c = 10.0
d1 = a*b + c
c += h
d2 = a*b + c
print('d1',d1)
print('d2',d2)
print('slope', (d2 - d1)/h)

#-----------------------------
class Value:
     def __init__(self,data, _children=(), _op='', label=''):
         self.data = data
         self._prev = set(_children)
         self._op = _op
         self.grad = 0.0
         self.label=label
     def __repr__(self):
         return f"Value(data={self.data})"
     def __add__(self,other):
         out = Value(self.data + other.data, (self,other), "+")
         return out
     def __mul__(self,other):
         out = Value(self.data * other.data, (self,other), "*")
         return out
         
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b
e.label = 'e'
d = e+c
d.label = 'd'

#add new
f = Value(-2.0, label='f')
L = d*f
L.label = 'L'

print(d)
print(d._prev)

# class for graph nodes:

from graphviz import Digraph

def trace(root):
    #builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format= 'svg', graph_attr={'rankdir': "LR"}) #: LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f}" % (n.label,n.data,n.grad ), shape='record')
        if n._op:
            #if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
            
    for n1, n2 in edges:
        #connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    dot.render('output_graph', view=True)
    return dot

# L = d*f
#dL/dd = ? f

L.grad = 1
f.grad = 4.0
d.grad = -2
c.grad = -2
e.grad = -2
#dl/dc = dl/dd * dd/dc = dl/dd * 1.0 = f = -2.0
#dL/dc = ? or dd/dc = 1.0 = dd/de (d = e + c)
#dl/de = -2, e = a*b 
#dl/da = (dl/de) * (de/da) = -2.b = 6
a.grad = -2.0 * -3.0
b.grad = -2.0 * 2.0

draw_dot(L)

# demo backpropgation:
def lol():
    h = 0.001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b
    e.label = 'e'
    d = e+c
    d.label = 'd'

    #add new
    f = Value(-2.0, label='f')
    L = d*f
    L.label = 'L'

    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b
    e.label = 'e'
    d = e+c
    d.label = 'd'

    #add new
    f = Value(-2.0, label='f')
    L = d*f
    L.label = 'L'

    L2 = L.data + h

    print((L2-L1)/h)

lol()

#increasing the value for increase L
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

draw_dot(L)

# neuron

