# The first (failed) prototype of the block classes. 
# Successful in the ability to compose funcitons, but the structure became to convoluted and unsuitable to manual slotting.
# No graphcis involved at this stage.

import Axis

class blockType():
    def __init__(self,opcode,operation):
        self.opcode = opcode
        self.operation = operation
        self.operands = len(operation.__code__.co_varnames)
        if isinstance(self,iterable):
            self.color = 'red'
        elif isinstance(self,predicate):
            self.color = 'orange'
        #elif isinstance(self,control):
        #    self.color = 'yellow'
        elif isinstance(self,letters):
            self.color = 'green'
        elif isinstance(self,arithmetic):
            self.color = 'blue'
        #elif isinstance(self,animation):
        #    self.color = 'purple'
        else:
            self.color = 'grey'

    def __call__(self,*operands):
        if len(operands) != self.operands:
            raise TypeError(f'{self.opcode} of {operands}')
        if type(operands[0]) == blockTemplate:
            return self.operation(*map(lambda x: x.__call__(),operands))
        return self.operation(*operands)

class predicate(blockType):
    def __init__(self,opcode,operation):
        super().__init__(opcode,operation)
        self.null = False

    def __call__(self,*operands):
        if type(operands) == bool:
            return operands
        if len(operands) != self.operands:
            raise TypeError(f'{self.opcode} of {operands}')
        return bool(self.operation(*operands))

class arithmetic(blockType):
    def __init__(self,opcode,operation,null=0):
        super().__init__(opcode,operation)
        self.null = null

class letters(blockType):
    def __init__(self,opcode,operation,null=''):
        super().__init__(opcode,operation)
        self.null = null

class iterable(blockType):
    def __init__(self,opcode,operation,null=None):
        super().__init__(opcode,operation)
        self.null = null

class blockTemplate():
    def __init__(self,blockType,*inputs):
        self.type = blockType
        self.inputs = tuple(*[inputs])
        self.missing = self.type.operands - len(self.inputs)
        self.filler = (self.type.null,) * self.missing
        self.inputs = self.inputs + self.filler
        self.code = self.type(*self.inputs)
        self.color = self.type.color

    def __repr__(self):
        return f'{self.type.opcode}{self.inputs}'.replace(',)',')')

    def __hash__(self):
        return(hash(self.inputs))

    def __call__(self):
        return self.code

functionDict = {}

def createFunction(name,expType,func,d):
    if name in d:
        raise NameError(f'{name} already exists')
    d[name] = blockTemplate(expType(name,func))

AND = predicate('and',lambda x,y: x and y)
OR = predicate('or',lambda x,y: x or y)
NOT = predicate('not',lambda x: not x)
NAND = predicate('nand',lambda x,y: NOT(AND(x,y)))
NOR = predicate('nor',lambda x,y: NOT(OR(x,y)))
XOR = predicate('xor',lambda x,y: AND(OR(x,y),NOT(AND(x,y))))
XNOR = predicate('xnor',lambda x,y: OR(NOR(x,y),AND(x,y)))

TRUE = predicate('true',lambda: True)
FALSE = predicate('false',lambda: False)

LT = predicate('<',lambda x,y: x < y)
LE = predicate('<=',lambda x,y: x <= y)
EQ = predicate('==',lambda x,y: x == y)
GE = predicate('>=',lambda x,y: x >= y)
GT = predicate('>',lambda x,y: x > y)

ADD = arithmetic('+',lambda x,y: x + y)
SUB = arithmetic('-',lambda x,y: x - y)
MULT = arithmetic('*',lambda x,y: x * y)
POW = arithmetic('**',lambda x,y: x ** y)
FDIV = arithmetic('/',lambda x,y: x / y)
IDIV = arithmetic('//',lambda x,y: x // y)
MOD = arithmetic('%',lambda x,y: x % y)
EXPR = arithmetic('EXPR',lambda x: x)
ROUND = arithmetic('ROUND',lambda x,y: round(x,y))

ODD = predicate('even',lambda x: EQ(MOD(x,2),1))
EVEN = predicate('odd',lambda x: EQ(MOD(x,2),0))

ALIST = iterable('list',lambda x: Axis.Axis(x))
TLIST = iterable('list',lambda x,y: Axis.Axis(x,y))
FLIST = iterable('list',lambda w,x,y,z: Axis.Axis(w,x,y,z))
ELIST = iterable('list',lambda s,t,u,v,w,x,y,z: Axis.Axis(s,t,u,v,w,x,y,z))
FIRST = iterable('first',lambda a: a.head())
RANGE = iterable('range',lambda a,b,c:Axis.Axis(*range(a,b,c)))
ALLBUTFIRST = iterable('all but first',lambda a: a.tail())
LAST = iterable('last',lambda a: a.last())
ALLBUTLAST = iterable('all but last',lambda a: a.blast())
SLICE = iterable('slice',lambda a,b,c,d: a[b:c:d])
LENGTH = iterable('length',lambda a: a.length)

A1 = blockTemplate(ALIST,1)
A2 = blockTemplate(TLIST,1,2)
A3 = blockTemplate(FLIST,1,2,3)
A4 = blockTemplate(FLIST,1,2,3,4)
A5 = blockTemplate(ELIST,1,2,3,4,5)
A6 = blockTemplate(ELIST,1,2,3,4,5,6)
A7 = blockTemplate(ELIST,1,2,3,4,5,6,7)
A8 = blockTemplate(ELIST,1,2,3,4,5,6,7,8)

I1 = blockTemplate(EXPR,1)
I2 = blockTemplate(EXPR,2)
I3 = blockTemplate(EXPR,3)

P1 = blockTemplate(AND,True,True)
P2 = blockTemplate(XOR,True,False)
P3 = blockTemplate(AND,P1,P2)

# __defaults__
