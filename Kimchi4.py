import copy, math, random, string
from tkinter import *
import Axis
from cmu_112_graphics import *

blockLibrary = {} # Dictionary of all known functions/atoms (used for defaults & drawing function library)
ioLibrary = {} # Dictionary of input atoms used for function genesis
maxVars = 16 # Number of pink variables used in sandbox mode

spiralCords = [] # Cordinates used to draw a spiral
for t in range(25):
    spiralCords.append(605+t*math.cos(t)/1.3)
    spiralCords.append(675+t*math.sin(t)/1.3)

def fillBlockWithInputs(f,i): # Parses through list of strings representing a function to add inputs
    if f in list(ioLibrary.keys()):
        return i[int(f[1])]
    if isinstance(f,str):
        return f
    return list(map(lambda x: fillBlockWithInputs(x,i),f))

def callBlockWithInputs(f): # Parses through a list of strings representing a function (with inputs) to call it
    if isinstance(f,str) and f in blockLibrary:
        return blockLibrary[f]()
    if not isinstance(f,list):
        return f
    func = blockLibrary[f[0]]
    try:
        if isinstance(func,ControlFunction):
            return func.value(f[1],callBlockWithInputs(f[2]))
        if isinstance(func,CustomFunction):
            return callBlockWithInputs(fillBlockWithInputs(f[0],list(map(lambda x: callBlockWithInputs(x),f[1:]))))
        return func.value(*map(lambda x: callBlockWithInputs(x), f[1:]))
    except Exception as e:
        TopLevel.globalException(e)
        try:
            return func()
        except:
            return None

def darkenColor(h): # Takes in a hex color and returns a darkened version
    conversion = {'f':'d','e':'c','d':'b','c':'a','b':'9','a':'8','9':'7',
    '8':'6','7':'5','6':'4','5':'3','4':'2','3':'1','2':'0','1':'0','0':'0'}
    out = '#'
    for i in h[1:]:
        out += conversion[i]
    return out

def chunkifyString(s,l): # Chops up a string into lines of length l
    out = []
    row = ''
    for w in s.split():
        if len(w) + len(row) > l:
            out.append(row)
            row = w
        else:
            row += ' ' + w
    out.append(row)
    return '\n'.join(out).strip()

def customError(e):
    return str(e)

def rgb2hex(r,g,b): # Returns a hexedecimal color string based on 3 rgb integers
    h1 = hex(r)[2:]
    h2 = hex(g)[2:]
    h3 = hex(b)[2:]
    if r < 16: h1 = '0' + h1
    if g < 16: h2 = '0' + h2
    if b < 16: h3 = '0' + h3
    return '#' + h1 + h2 + h3

def ifThenElse(condition,option1,option2): # If-else logic
    if condition:
        return option1
    return option2

def switch(value,cases,outcomes): # Switch statements (to simplify excessive if-else nesting)
    for i in range(len(cases)):
        if value == cases[i]:
            return outcomes[i]
    return blockLibrary['Null']

def minimize(f,l): # Loop over a list and pick out the item with the smallest output to some function
    maxInput = None
    maxOutput = float('inf')
    for i in l:
        hof = replaceLambda(f,i)
        if hof < maxOutput:
            maxInput = i
            maxOutput = hof
    return maxInput

def optimize(f,l): # Loop over a list and pick out the item with the largest output to some function
    maxInput = None
    maxOutput = float('-inf')
    for i in l:
        hof = replaceLambda(f,i)
        if hof > maxOutput:
            maxInput = i
            maxOutput = hof
    return maxInput

def findFirst(f,l): # Loop over a list and pick out the first item that returns true for some function
    for i in l:
        if replaceLambda(f,i):
            return i
    return None

def get(t,d,default): # Treat a list of 2-item lists as a dicitonary with the .get() method
    for i in d:
        if i[0] == t:
            return i[1]
    return default

def kSort(f,l): # Sorting HOF
    out = []
    for i in l:
        j = len(out)
        k = replaceLambda(f,i)
        while j > 0:
            if k < replaceLambda(f,out[j-1]):
                j -= 1
            else:
                break
        out.insert(j,i)
    return Axis.Axis(*out)

def kMap(f,l): # Mapping HOF
    out = []
    for i in l:
        out.append(replaceLambda(f,i))
    return Axis.Axis(*out)

def kFilter(f,l): # Filtering HOF
    out = []
    for i in l:
        if replaceLambda(f,i):
            out.append(i)
    return Axis.Axis(*out)

def kCombine(f,l): # Combination HOF
    out = l[0]
    for i in l[1:]:
        out = replaceLambda(f,out,i)
    return out

def replaceLambda(f,i,m=None): # Replaces all lambda atoms with the desired input (modified to work with string parser)
    if f == chr(955) or (isinstance(f,Atom) and f.name == chr(955)):
        return i
    elif f == chr(956) or (isinstance(f,Atom) and f.name == chr(956)):
        return m
    elif isinstance(f,Function):
        return f.value(*map(lambda x: replaceLambda(x,i,m),f.operands))
    elif isinstance(f,list):
        return blockLibrary[f[0]].value(*map(lambda x: replaceLambda(x,i,m),f[1:]))
    elif isinstance(f,str):
        return blockLibrary[f].value
    return f()

class Dragger(object): # Class of all draggable blocks

    cloneList = [] # List of all block clones (in sandbox mode)
    cloneList2 = [] # List of all block clones (in function mode)
    fontSize = 17
    targetSize = 10
    
    def __bool__(self): # If treated as a boolean, return boolean of value
        return bool(self())

    def touching(self,x,y,app): # Is this block being touched
        if self.cloneable: # If cloneable, create a clone (and adjust for scrolling)
            if (x-self.x)**2 + (y+app.scrollY-self.y)**2 < 100:
                if isinstance(self,Atom):
                    if type(self) == VarAtom:
                        return VarAtom(self.value,x,y+app.scrollY2,False,self.name)
                    elif type(self) == IOAtom:
                        if int(self.name[1]) >= GenesisFunction.using:
                            return
                        return IOAtom(self.value,x,y+app.scrollY2,False,self.name)
                    return Atom(self.value,x,y+app.scrollY2,False,self.name)
                if type(self) == CustomFunction:
                    return eval(self.fType)(self.name,self.value,x,y,False,self.defaults,[None for i in range(len(self.operands))],None,self.color)
                return eval(self.fType)(self.name,self.value,x,y,False,self.defaults,[None for i in range(len(self.operands))],None)
        elif (x-self.x)**2 + (y+app.scrollY2-self.y)**2 < 100: # Else, just move it (and bring to top)
            if self.name != 'Set Variables':
                if type(app) == Sandbox:
                    Dragger.cloneList.remove(self)
                    Dragger.cloneList.append(self)
                elif type(app) == CreationMode:
                    Dragger.cloneList2.remove(self)
                    Dragger.cloneList2.append(self)
            return self

    def __eq__(self,other):
        return type(self) == type(other) and self() == other() and self.x == other.x and self.y == other.y

    def FuncToString(self):
        if isinstance(self,Atom):
            return self.name
        return [self.name] + list(map(lambda x: x.FuncToString(),self.operands))

class Function(Dragger): # Class of all functions
    
    def __init__(self,fType,name,value,x,y,cloneable,defaults,slots,parent=None):
        self.fType, self.name, self.value, self.x, self.y = fType, name, value, x, y
        self.cloneable, self.defaults, self.slots = cloneable, defaults, slots
        self.parent = parent
        self.getDimensions()
        self.operands = slots[:]
        self.getOperands()
        self.buffer = 5
        if self.cloneable: blockLibrary[self.name] = self
        elif TopLevel.curMode == 2: Dragger.cloneList2.append(self)
        else: Dragger.cloneList.append(self)

    def getDimensions(self): # Update dimensions as things are dragged in/out
        self.xMin = Dragger.targetSize + 10 + len(self.name) * Dragger.fontSize/2.3
        self.xMax = max([self.xMin]+list(map(lambda a: self.xMin if a == None else a.xMax + Dragger.targetSize + 5, self.slots)))
        self.yMin = 2 * Dragger.targetSize
        self.yCords = [0]
        for i in range(len(self.slots)):
            if i == 0: self.yCords.append(self.yMin*1.1)
            elif self.slots[i-1] == None: self.yCords.append(self.yCords[-1]+self.yMin*1.1)
            else: self.yCords.append(self.yCords[-1]+self.slots[i-1].ySize)
        self.ySize = self.yCords[-1]
        if self.slots[-1] != None:
            self.ySize += self.slots[-1].ySize
        else:
            self.ySize += self.yMin * 1.1
        if self.parent != None:
            self.parent.getDimensions()

    def getOperands(self): # Update operands according to if slots are empty or not
        for i in range(len(self.slots)):
            if self.slots[i] == None: self.operands[i] = self.defaults[i]
            else: self.operands[i] = self.slots[i]

    def __call__(self): # If being treated like a function, call the lambda expression
        try:
            val = self.value(*map(lambda x: x(), self.operands))
            if isinstance(val,float):
                return round(val,8)
            return val
        except Exception as e: # If inputs are invalid, return the defualt and print an error
            TopLevel.globalException(e)
            return self.value(*map(lambda x: x(), self.defaults))

    def exception(self):
        try:
            self.value(*map(lambda x: x(), self.operands))
            return False
        except:
            return True

    def __repr__(self): # How to print for debugging:
        return f'{self.name} ({", ".join(map(lambda x: str(x),self.operands))}) => {str(self())}'

    def drawLibraryFunction(self,canvas,app): # Draws the cloneable functions in the library
        clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = 150, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x,y0-app.scrollY,x1,y1-app.scrollY,fill=self.color,width=0)
        canvas.create_oval(x0,y0-app.scrollY,x2,y1-app.scrollY,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY,anchor='w',text=self.name,font=f'Times {Dragger.fontSize}')

    def drawFunction(self,canvas,app): # Draws the cloned functions
        if self == app.holding:
            clickColor = self.clickColor
        else: clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + self.xMax + Dragger.targetSize, self.y + Dragger.targetSize
        y2 = y1  + self.ySize - self.yMin
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x+2,y0-app.scrollY2+2,x1-2,y2-app.scrollY2-2,fill=self.color,width=2,outline=darkenColor(self.color))
        canvas.create_oval(x0,y0-app.scrollY2,x2,y1-app.scrollY2,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY2,anchor='w',text=self.name,font=f'Times {Dragger.fontSize}')
        for i in range(len(self.operands)):
            if self.slots[i] == None:
                yCord = self.yCords[i+1] - self.yMin*1.1
                canvas.create_oval(self.x+self.buffer,y1+yCord-app.scrollY2,self.x+self.buffer+Dragger.targetSize*2,y1+yCord+Dragger.targetSize*2-app.scrollY2,fill='White',width=0)
            elif isinstance(self.operands[i],Atom):
                self.operands[i].drawAtom(canvas,app)
            elif isinstance(self.operands[i],Function):
                self.operands[i].drawFunction(canvas,app)

    def genesis(s,x,y): # Generates a new function from a string
        parts = tuple(s.split(' | '))
        fType, name, func, defaultNames = parts
        func = eval(func)
        defaults = list(map(lambda x: blockLibrary[x], defaultNames.split(' ')))
        for t in 'ArithmeticFunction','LetterFunction','IterableFunction','PredicateFunction','ControlFunction':
            if fType == t:
                return eval(f'{t}')(name,func,x,y,True,defaults,[None for i in range(len(defaults))])

    def insideSlot(self,x,y): # Is a given x,y coordinate inside one of a function's slots
        if abs(x-self.x-Dragger.targetSize-self.buffer) >= Dragger.targetSize:
            return
        for r in range(1,len(self.yCords)):
            i = self.yCords[r]
            if abs(i+self.y-y) < Dragger.targetSize:
                return r-1

    def slotify(self,allowUp=True): # Adjust x & y coordiantes of all children in refrence to parent
        for i in range(1,len(self.yCords)):
            if self.slots[i-1] != None:
                self.slots[i-1].y = self.y + self.yCords[i] - 3
                self.slots[i-1].x = self.x + Dragger.targetSize + self.buffer
                if isinstance(self.slots[i-1],Function):
                    self.slots[i-1].slotify(allowUp)
        if self.parent != None and allowUp:
            self.parent.slotify(False)
        self.getDimensions()
        self.getOperands()

    def killChildren(self): # Recursively delete all children if the parent is discarded
        for child in self.slots:
            if isinstance(child,Function):
                child.killChildren()
            if isinstance(child,Dragger):
                if child in Dragger.cloneList:
                    Dragger.cloneList.remove(child)
                elif child in Dragger.cloneList2:
                    Dragger.cloneList2.remove(child)

class ArithmeticFunction(Function):
    color = '#49bcdf'
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('ArithmeticFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#49bcdf'

class LetterFunction(Function):
    color = '#94df49'
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('LetterFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#94df49'

class PredicateFunction(Function):
    color = '#9079f6'
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('PredicateFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#9079f6'

class IterableFunction(Function):
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('IterableFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#ea5639'

class ControlFunction(Function):
    color = '#ea5639'
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('ControlFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#e1ae5b'

    def exception(self):
        try:
            self.value(self.operands[0],*map(lambda x: x(),self.operands[1:]))
            return False
        except:
            return True

    def __call__(self): # If being treated like a function, call the lambda expression
        try:
            val = self.value(self.operands[0],*map(lambda x: x(),self.operands[1:]))
            if isinstance(val,float):
                return round(val,8)
            return val
        except Exception as e: # If inputs are invalid, return the defualt and print an error
            TopLevel.globalException(e)
            return self.value(*map(lambda x: x(), self.defaults))

class VarSetFunction(Function):
    color = '#e85f64'
    def __init__(self,name):
        super().__init__('VarSetFunction',name,None,165,20,False,TopLevel.Defaults,TopLevel.Vars,None)
        self.color = '#e85f64'
        self.buffer = 35

    def __call__(self): # If being called, do nothing
        return

    def drawFunction(self,canvas,app): # Draws the variable setter
        self.xMax = max([self.xMin]+list(map(lambda a: a.xMax + Dragger.targetSize + self.buffer if isinstance(a,Dragger) else 0, self.slots)))
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + self.xMax + Dragger.targetSize, self.y + Dragger.targetSize
        y2 = y1  + self.ySize - self.yMin
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x+2,y0-app.scrollY2+2,x1-2,y2-app.scrollY2-2,fill=self.color,width=2,outline=darkenColor(self.color))
        canvas.create_text(x2-3,self.y-app.scrollY2,anchor='w',text=self.name,font=f'Times {Dragger.fontSize}')
        for i in range(len(self.operands)):
            canvas.create_text(self.x+self.buffer/2,self.y+self.yCords[i+1]-app.scrollY2,text=f'V{i}',font=f'Times {Dragger.fontSize-2}')
            if self.slots[i] == None:
                yCord = self.yCords[i+1] - self.yMin*1.1 -app.scrollY2
                canvas.create_oval(self.x+self.buffer,y1+yCord,self.x+self.buffer+Dragger.targetSize*2,y1+yCord+Dragger.targetSize*2,fill='White',width=0)
            elif isinstance(self.operands[i],Atom):
                self.operands[i].drawAtom(canvas,app)
            elif isinstance(self.operands[i],Function):
                self.operands[i].drawFunction(canvas,app)

class GenesisFunction(Function):
    color = '#e85f64'
    curName = ''
    using = 0

    def __init__(self,name):
        super().__init__('GenesisFunction',name,None,165,20,False,[blockLibrary['Null']],[None],None)
        self.color = '#e85f64'
        self.buffer = 5

    def __call__(self): # If being called, do nothing
        return

    def drawFunction(self,canvas,app): # Draws the variable setter
        self.xMax = max([self.xMin]+list(map(lambda a: a.xMax + Dragger.targetSize + self.buffer if isinstance(a,Dragger) else 0, self.slots)))
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + self.xMax + Dragger.targetSize, self.y + Dragger.targetSize
        y2 = y1  + self.ySize - self.yMin
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x+2,y0-app.scrollY2+2,x1-2,y2-app.scrollY2-2,fill=GenesisFunction.color,width=2,outline=darkenColor(GenesisFunction.color))
        canvas.create_text(x2-3,self.y-app.scrollY2,anchor='w',text=GenesisFunction.curName,font=f'Times {Dragger.fontSize}')
        for i in range(len(self.operands)):
            if self.slots[i] == None:
                yCord = self.yCords[i+1] - self.yMin*1.1 -app.scrollY2
                canvas.create_oval(self.x+self.buffer,y1+yCord,self.x+self.buffer+Dragger.targetSize*2,y1+yCord+Dragger.targetSize*2,fill='White',width=0)
            elif isinstance(self.operands[i],Atom):
                self.operands[i].drawAtom(canvas,app)
            elif isinstance(self.operands[i],Function):
                self.operands[i].drawFunction(canvas,app)

class CustomFunction(Function):
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None,color=None):
        super().__init__('CustomFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = color
        self.dataString = ''

    def __call__(self):
        try:
            if self.name == CreationMode.frankenstein.name:
                me = TopLevel.FuncSlots.slots[0].FuncToString()
            else:
                me = blockLibrary[self.name].dataString
            val = callBlockWithInputs(fillBlockWithInputs(me,list(map(lambda x: x(), self.slots))))
            return val
        except:
            return None

class Atom(Dragger): # Class of all atoms
    def __init__(self,value,x,y,cloneable,name,parent=None):
        self.value, self.x, self.y, self.cloneable, self.name, self.parent = value, x, y, cloneable, name, parent
        self.xMin = self.xMax = Dragger.targetSize + 10 + len(self.name) * Dragger.fontSize/2.3
        self.yMin = self.ySize = 2 * Dragger.targetSize
        self.getColor()
        if self.cloneable: 
            if type(self) == IOAtom:
                ioLibrary[self.name] = self
            else:
                blockLibrary[self.name] = self
        elif TopLevel.curMode == 2: Dragger.cloneList2.append(self)
        else: Dragger.cloneList.append(self)
        
    def getColor(self): # Does exactly what is sounds like
        if type(self) == IOAtom or (self.name[0] == 'V' and len(self.name)>1): self.color = '#e85f64'
        elif isinstance(self.value,(int,float)):
            if isinstance(self.value,bool): self.color = '#9079f6'
            else: self.color = '#49bcdf'
        elif isinstance(self.value,str): self.color = '#94df49'
        elif isinstance(self.value,Axis.Axis): self.color = '#ea5639'
        else: self.color = '#b7abbf'

    def __call__(self): # If treated as a function, return the value
        try:
            return self.value
        except:
            return None

    def __repr__(self): # How to print for debugging
        return str(self())

    def drawLibraryAtom(self,canvas,app): # Draws the cloneable atoms in the library
        clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = 150, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x,y0-app.scrollY,x1,y1-app.scrollY,fill=self.color,width=0)
        canvas.create_oval(x0,y0-app.scrollY,x2,y1-app.scrollY,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY,anchor='w',text=self.name,font=f'Times {Dragger.fontSize}')

    def drawAtom(self,canvas,app): # Draws the cloned atoms
        if self == app.holding: clickColor = '#2e9121' # Green circle if currently being held
        else: clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + Dragger.targetSize + self.xMin, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x+2,y0-app.scrollY2+2,x1-2,y1-app.scrollY2-2,fill=self.color,width=2,outline=darkenColor(self.color))
        canvas.create_oval(x0,y0-app.scrollY2,x2,y1-app.scrollY2,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY2,anchor='w',text=self.name,font=f'Times {Dragger.fontSize}')

class VarAtom(Atom): # Subclass of Atom to handle variable atoms
    def __init__(self,value,x,y,cloneable,name,parent=None):
        self.x, self.y, self.cloneable, self.name, self.parent = x, y, cloneable, name, parent
        self.value = self()
        self.xMin = self.xMax = Dragger.targetSize + 10 + len(self.name) * Dragger.fontSize/2.3
        self.yMin = self.ySize = 2 * Dragger.targetSize
        self.getColor()
        if cloneable: blockLibrary[self.name] = self
        else: Dragger.cloneList.append(self)

    def __call__(self): # Try-except to prevent circular logic (V2 = V1 but V1 = V2)
        try:
            return TopLevel.Vars[int(self.name[1:])]()
        except:
            return None

class IOAtom(Atom):

    def drawAtom(self,canvas,app): # Draws the cloned atoms
        if self == app.holding: clickColor = '#2e9121' # Green circle if currently being held
        else: clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + Dragger.targetSize + self.xMin, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x+2,y0-app.scrollY2+2,x1-2,y1-app.scrollY2-2,fill=self.color,width=2,outline=darkenColor(self.color))
        canvas.create_oval(x0,y0-app.scrollY2,x2,y1-app.scrollY2,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY2,anchor='w',text=self.name,font=f'Times {Dragger.fontSize}')

    def getColor(self): # Pink if useable, grey if not (a 3-input function can only use inputs 0, 1 & 2)
        if int(self.name[1]) < GenesisFunction.using: 
            self.color = '#e85f64'
        else:
            self.color = '#b7abbf'

class Sandbox(Mode): # Mode with the sandbox for playing with Atoms & Functions
    def appStarted(self):
        self.holding = [None]
        self.scrollY = 5
        self.scrollY2 = 0
        self.scrollY3 = 0
        self.scrollMax = len(blockLibrary) * Dragger.targetSize * 1.5
        self.scrollMax2 = 100
        self.scrollMax3 = 100
        self.scrollHeight = self.height
        self.scrolling = False
        self.scrolling2 = False
        self.scrolling3 = False
        self.VarString = ''

    def modeActivated(self): # Scrap the graphics file canvas so that I can handle exceptions myself
        self.scrollMax = sorted(list(blockLibrary.values()),key=lambda x: x.y)[-1].y+2*Dragger.targetSize - 800
        self.app._canvas.pack_forget()
        self.canvas = Canvas(self.app._root,width=800,height=800)
        self.canvas.config(width=800,height=800)
        self.canvas.pack()
        TopLevel.globalCanvas = self.canvas
        TopLevel.curMode = 1

    def modeDeactivated(self): # When deactivated, re-impose the graphics-file canvas
        self.canvas.pack_forget()
        self.app._canvas.pack()

    def pressingButtons(self,x,y):
        y0, y1, y2 = 675, 725, 775
        y0, y1, y2, y3 = 650, 700, 750, 800
        options = [self.app.splashMode,self.app.splashMode,self.app.splashMode]
        yVals = [675, 725, 775]
        for i in range(3):
            if (x > 590 or ((x-590)**2 + (y-yVals[i])**2) < 625) and x < 630 and y < yVals[i]+25 and y > yVals[i]-25:
                if i == 1:
                    self.getFunctionInfo()
                else:
                    self.app.setActiveMode(options[i])

    def getFunctionInfo(self):
        if not CreationMode.done:
            self.app.setActiveMode(self.app.creationMode)
        else:
            fName = self.getUserInput('What is the function\'s name?')
            if fName == None or fName == '' or len(fName) > 14 or fName in map(lambda x: x.name, blockLibrary.values()):
                nextStep = self.getUserInput(f'"{fName}" is not a valid name.\nType exit to quit.\nType anything else to try agian.')
                if nextStep != 'exit':
                    self.getFunctionInfo()
            else:
                fType = self.getUserInput(f'What type of function is {fName}?')
                if fType == None or fType.lower() not in ['letter','arithmetic','iterable','predicate']:
                    if fType == 'control':
                        nextStep = self.getUserInput(f'You cannot make control functions.\nType exit to quit.\nType anything else to try agian.')
                    else:
                        nextStep = self.getUserInput(f'"{fType}" is not a valid function type.\nValid types: arithmetic, predicate, letter, iterable\nType exit to quit.\nType anything else to try agian.')
                    if nextStep != 'exit':
                        self.getFunctionInfo()
                else:
                    fInpt = self.getUserInput(f'How many inputs does {fName} have?')
                    if fInpt not in map(lambda x: str(x), range(10)):
                        nextStep = self.getUserInput(f'"{fInpt}" is not a valid number of inputs.\nType exit to quit.\nType anything else to try agian.')
                        if nextStep != 'exit':
                            self.getFunctionInfo()
                    else:
                        self.app.funcStrings = (fName,fType,fInpt)
                        GenesisFunction.curName = fName
                        GenesisFunction.color = eval(f'{fType.lower().title()}Function.color')
                        GenesisFunction.using = int(fInpt)
                        for block in ioLibrary.values():
                            block.getColor()
                        CreationMode.done = False
                        CreationMode.megaGenesis(*self.app.funcStrings)
                        TopLevel.FuncSlots.slots = [None]
                        TopLevel.FuncSlots.getDimensions()
                        Dragger.cloneList2 = []
                        self.app.setActiveMode(self.app.creationMode)

    def mousePressed(self,event): # Check for scrolling & dragging Atoms/Functions
        self.pressingButtons(event.x,event.y)
        if event.x < 10:
            self.scrolling = True
            return
        if event.x < 640 and event.x > 630:
            self.scrolling2 = True
            return
        if event.x > 790:
            self.scrolling3 = True
            return
        for i in list(blockLibrary.values()) + Dragger.cloneList:
            if i == None or i.name == 'Create Function':
                continue
            contact = i.touching(event.x,event.y,self)
            if contact != None:
                if contact.name == 'Set Variables':
                    return
                self.holding = contact
                if isinstance(contact,Function) and contact.exception():
                    contact.clickColor = '#f2e422' # Yellow circle if causing an error
                else:
                    contact.clickColor = '#2e9121' # Green circle if currently being held
                if contact.name == CreationMode.frankenstein.name:
                    print(contact())
                if self.holding.parent != None:
                    for i in range(len(self.holding.parent.slots)):
                        if self.holding.parent.slots[i] == self.holding:
                            if self.holding.parent.name == 'Set Variables':
                                TopLevel.Vars[i] = None
                                blockLibrary[f'V{i}'].value = None
                                for k in Dragger.cloneList:
                                    for j in range(maxVars):
                                        if k.name == 'V'+str(j):
                                            try:
                                                k.value = TopLevel.Vars[j]
                                            except:
                                                pass
                            self.holding.parent.slots[i] = None
                    self.holding.parent.slotify()
                return

    def mouseDragged(self,event): # Move an Atom/Funciton if it is being dragged, or adjust scrollbar
        if self.scrolling:
            self.scrollY = max(0,event.y)/self.height * self.scrollMax
        if self.scrolling2:
            self.scrollMax2 = min(800,TopLevel.VariableSlots.ySize)
            self.scrollY2 = max(0,event.y)/self.height * self.scrollMax2
        if self.scrolling3:
            self.scrollMax3 = min(670,15*Sandbox.VarString.count('\n'))
            self.scrollY3 = min(max(0,event.y),670)/self.height * self.scrollMax3
        if self.holding == [None] or self.holding.name == 'Set Variables':
            return
        self.holding.x = event.x
        self.holding.y = event.y + self.scrollY2
        if isinstance(self.holding,Function):
            self.holding.slotify()

    def mouseReleased(self,event): # Stop scrolling & release dragged Atoms/Funcitons
        self.scrolling = False
        self.scrolling2 = False
        self.scrolling3 = False
        if self.holding == [None] or self.holding.name == 'Set Variables':
            self.holding = [None]
            return
        self.holding.x = event.x
        self.holding.y = event.y + self.scrollY2
        if isinstance(self.holding,Function):
            self.holding.slotify()
        xEdge = event.x + self.holding.xMax/2
        if event.x - Dragger.targetSize < 150 or xEdge > self.width*0.8:
            Dragger.cloneList.remove(self.holding)
            if isinstance(self.holding,Function):
                self.holding.killChildren()
        else:
            for i in Dragger.cloneList + [TopLevel.VariableSlots]:
                if isinstance(i,Function):
                    if i.name == 'Create Function':
                        continue
                    slot = i.insideSlot(event.x,event.y+self.scrollY2)
                    if slot != None:
                        i.slots[slot] = self.holding
                        self.holding.parent = i
                        i.slotify()
                        if i.name == 'Set Variables':
                            TopLevel.Vars[slot] = self.holding
                            blockLibrary[f'V{slot}'].value = self.holding.value
                            for i in Dragger.cloneList:
                                for j in range(maxVars):
                                    if i.name == 'V'+str(j):
                                        try:
                                            i.value = TopLevel.Vars[j]
                                        except:
                                            pass
                        break
        self.holding = [None]

    def drawLibraryScroll(self,canvas): # Draw the scrollbar(s)
        y2 = min(max(10,self.height * self.scrollY / self.scrollMax),790)
        y3 = min(max(5,self.height * self.scrollY2 / self.scrollMax2),790)
        y4 = min(max(5,self.height * self.scrollY3 / self.scrollMax3),660)
        canvas.create_rectangle(0,0,15,self.height,fill='#d1d1d2',width=0)
        canvas.create_oval(4,y2-7,14,y2+7,fill='#a8a7a9',width=0)
        canvas.create_rectangle(630,0,640,self.height,fill='#d1d1d2',width=0)
        canvas.create_oval(632,y3-3,638,y3+3,fill='#a8a7a9',width=0)
        canvas.create_rectangle(790,0,800,self.height*0.8375,fill='#d1d1d2',width=0)
        canvas.create_oval(790,y4-3,797,y4+3,fill='#a8a7a9',width=0)

    def drawEvaluator(self,canvas):
        x0 = self.width * 0.81
        y0 = 10 - self.scrollY3
        out = ['Variable Evaluator:']
        for i in range(maxVars):
            v = TopLevel.Vars[i]
            if v == None or v() == None:
                out.append(f'V{i} = Null')
            else:
                if isinstance(v(),float):
                    out.append(chunkifyString(f'V{i} = {round(v(),8)}',28))
                else:
                    out.append(chunkifyString(f'V{i} = {v()}',28))
        Sandbox.VarString = '\n'.join(out)
        canvas.create_text(x0,y0,anchor='nw',text=Sandbox.VarString,fill='Green',font='Times 10')

    def drawButtons(self,canvas):
        r = 25
        x0, x1 = 590, 630
        y0, y1, y2, y3 = 650, 700, 750, 800
        c1A, c2A, c3A = '#ad4040', '#67ad4e', '#40589c'
        c1B, c2B, c3B = darkenColor(c1A), darkenColor(c2A), darkenColor(c3A),
        canvas.create_rectangle(x0,y0,x1,y1,width=0,fill=c1B) # Draw animation-engine button
        canvas.create_oval(x0-r/1.5,y0,x0+r/1.5,y1,width=0,fill=c1B)
        canvas.create_rectangle(x0-2,y0+9,x1-9,y1-9,width=4,outline=c1A)
        canvas.create_line(*spiralCords,width=2,fill=c1A,smooth=True)
        canvas.create_rectangle(x0,y1,x1,y2,width=0,fill=c2B) # Draw function-cretor button
        canvas.create_oval(x0-r/1.5,y1,x0+r/1.5,y2,width=0,fill=c2B)
        xA, yA = (x0+x1-12)/2, y1 + 6
        xB, yB = x0 - 6, (y1 + y2)/2
        xC, yC = x1 - 6, y2 - 6
        canvas.create_polygon(xB,yB-5,xB,yB+5,xA-5,yB+5,xA-5,yC,xA+5,yC,xA+5,yB+5,xC,yB+5,
        xC,yB-5,xA+5,yB-5,xA+5,yA,xA-5,yA,xA-5,yB-5,width=0,fill=c2A,smooth=True)
        canvas.create_rectangle(x0,y2,x1,y3,width=0,fill=c3B) # Draw home button
        canvas.create_oval(x0-r/1.5,y2,x0+r/1.5,y3,width=0,fill=c3B)
        canvas.create_polygon(x0,y2+20,x0,y3-10,x1-15,y3-10,x1-15,y2+20,xA,y2+10,width=0,fill=c3A)

    def drawLibraryScroll(self,canvas): # Draw the scrollbar(s)
        y2 = min(max(10,self.height * self.scrollY / self.scrollMax),790)
        y3 = min(max(5,self.height * self.scrollY2 / self.scrollMax2),790)
        canvas.create_rectangle(0,0,15,self.height,fill='#d1d1d2',width=0)
        canvas.create_oval(4,y2-7,14,y2+7,fill='#a8a7a9',width=0)
        canvas.create_rectangle(630,0,640,self.height,fill='#d1d1d2',width=0)
        canvas.create_oval(632,y3-3,638,y3+3,fill='#a8a7a9',width=0)

    def redrawAll(self,canvas): # Manually implement animation using my own packed canvas
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(150,0,self.width,self.height,fill='Black',width=0)
        self.canvas.create_rectangle(1+self.width*0.8,self.height,self.width-2,self.height-129,fill='Black',width=3,outline='White')
        self.drawLibraryScroll(self.canvas)
        self.drawEvaluator(self.canvas)
        self.drawButtons(self.canvas)
        TopLevel.VariableSlots.drawFunction(self.canvas,self)
        for book in blockLibrary.values():
            if isinstance(book,Atom):
                book.drawLibraryAtom(self.canvas,self)
            else:
                book.drawLibraryFunction(self.canvas,self)
        for clone in Dragger.cloneList:
            if clone.name == 'Create Function':
                continue
            if isinstance(clone,Atom):
                clone.drawAtom(self.canvas,self)
            else:
                clone.drawFunction(self.canvas,self)

class CreationMode(Sandbox):

    done = True
    frankenstein = None

    @staticmethod
    def megaGenesis(name,typ,inputs):
        CreationMode.frankenstein = CustomFunction(name,None,25,sorted(list(blockLibrary.values()),key=lambda x: x.y)[-1].y+2*Dragger.targetSize,
        True,[blockLibrary['Null'] for i in range(int(inputs))],[None for i in range(int(inputs))],None,(eval(typ.lower().title()+'Function').color))

    def appStarted(self):
        self.holding = [None]
        self.scrollY = 5
        self.scrollY2 = 0
        self.scrollMax = len(blockLibrary) * Dragger.targetSize * 1.5 - 120
        self.scrollMax2 = 100
        self.scrollHeight = self.height
        self.scrolling = False
        self.scrolling2 = False
        self.fStrings = self.app.funcStrings

    def modeActivated(self): # Scrap the graphics file canvas so that I can handle exceptions myself
        self.scrollMax = sorted(list(blockLibrary.values()),key=lambda x: x.y)[-1].y+2*Dragger.targetSize - 920
        self.app._canvas.pack_forget()
        self.canvas = Canvas(self.app._root,width=800,height=800)
        self.canvas.config(width=800,height=800)
        self.canvas.pack()
        TopLevel.globalCanvas = self.canvas
        TopLevel.curMode = 2
        for i in blockLibrary.values():
            i.y -= 120

    def modeDeactivated(self): # When deactivated, re-impose the graphics-file canvas
        self.canvas.pack_forget()
        self.app._canvas.pack()
        for i in blockLibrary.values():
            i.y += 120

    def pressingButtons(self,x,y):
        y0, y1, y2 = 675, 725, 775
        y0, y1, y2, y3 = 650, 700, 750, 800
        options = [self.app.splashMode,self.app.sandboxMode,self.app.splashMode]
        yVals = [675, 725, 775]
        for i in range(3):
            if (x > 590 or ((x-590)**2 + (y-yVals[i])**2) < 625) and x < 630 and y < yVals[i]+25 and y > yVals[i]-25:
                if i == 1:
                    CreationMode.frankenstein.dataString = TopLevel.FuncSlots.slots[0].FuncToString()
                    CreationMode.done = True
                self.app.setActiveMode(options[i])

    def mousePressed(self,event): # Check for scrolling & dragging Atoms/Functions
        self.pressingButtons(event.x,event.y)
        if event.x < 10:
            self.scrolling = True
            return
        if event.x < 640 and event.x > 630:
            self.scrolling2 = True
            return
        for i in list(blockLibrary.values()) + list(ioLibrary.values()) + Dragger.cloneList2:
            if i == None or type(i) == VarAtom:
                continue
            contact = i.touching(event.x,event.y,self)
            if contact != None:
                self.holding = contact
                if contact.name == CreationMode.frankenstein.name:
                    print(contact())
                if isinstance(contact,Function) and contact.exception():
                    contact.clickColor = '#f2e422' # Yellow circle if causing an error
                else:
                    contact.clickColor = '#2e9121' # Green circle if currently being held
                if self.holding.parent != None:
                    for i in range(len(self.holding.parent.slots)):
                        if self.holding.parent.slots[i] == self.holding:
                            self.holding.parent.slots[i] = None
                    self.holding.parent.slotify()
                return

    def mouseDragged(self,event): # Move an Atom/Funciton if it is being dragged, or adjust scrollbar
        if self.scrolling:
            self.scrollY = max(0,event.y)/self.height * self.scrollMax
        if self.scrolling2:
            self.scrollMax2 = min(800,TopLevel.VariableSlots.ySize)
            self.scrollY2 = max(0,event.y)/self.height * self.scrollMax2
        if self.holding == [None]:
            return
        self.holding.x = event.x
        self.holding.y = event.y + self.scrollY2
        if isinstance(self.holding,Function):
            self.holding.slotify()

    def mouseReleased(self,event): # Stop scrolling & release dragged Atoms/Funcitons
        self.scrolling = False
        self.scrolling2 = False
        if self.holding == [None]:
            self.holding = [None]
            return
        self.holding.x = event.x
        self.holding.y = event.y + self.scrollY2
        if isinstance(self.holding,Function):
            self.holding.slotify()
        xEdge = event.x + self.holding.xMax/2
        if event.x - Dragger.targetSize < 150 or xEdge > self.width*0.8:
            Dragger.cloneList2.remove(self.holding)
            if isinstance(self.holding,Function):
                self.holding.killChildren()
        else:
            for i in Dragger.cloneList2 + [TopLevel.FuncSlots]:
                if isinstance(i,Function):
                    slot = i.insideSlot(event.x,event.y+self.scrollY2)
                    if slot != None:
                        i.slots[slot] = self.holding
                        self.holding.parent = i
                        i.slotify()
                        break
        self.holding = [None]

    def drawEvaluator(self,canvas):
        x0 = self.width * 0.81
        y0 = 10
        out = ['Function Creator:']
        out.append(f'Name: {self.fStrings[0]}')
        out.append(f'Type: {self.fStrings[1]}')
        out.append(f'Inputs: {self.fStrings[2]}')
        Sandbox.VarString = '\n'.join(out)
        canvas.create_text(x0,y0,anchor='nw',text=Sandbox.VarString,fill='Green',font='Times 10')

    def redrawAll(self,canvas): # Manually implement animation using my own packed canvas
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(150,0,self.width,self.height,fill='Black',width=0)
        self.canvas.create_rectangle(1+self.width*0.8,self.height,self.width-2,self.height-129,fill='Black',width=3,outline='White')
        self.drawLibraryScroll(self.canvas)
        self.drawEvaluator(self.canvas)
        self.drawButtons(self.canvas)
        TopLevel.FuncSlots.drawFunction(self.canvas,self)
        for book in list(ioLibrary.values()) + list(blockLibrary.values()):
            if isinstance(book,Atom):
                if type(book) != VarAtom:
                    book.drawLibraryAtom(self.canvas,self)
            else:
                book.drawLibraryFunction(self.canvas,self)
        for clone in Dragger.cloneList2:
            if isinstance(clone,Atom):
                clone.drawAtom(self.canvas,self)
            else:
                clone.drawFunction(self.canvas,self)

class SplashMode(Mode): # The colorful splash screen mode
    def modeActivated(self): # When activated, discard the graphics-file canvas
        self.app._canvas.pack_forget()
        self.canvas = Canvas(self.app._root,width=800,height=800)
        self.canvas.config(width=800,height=800)
        self.canvas.pack()
        TopLevel.globalCanvas = self.canvas

    def appStarted(self):
        self.spin = 0
        self.timerDelay = 5
        self.squares = [self.polarize(400,250,self.hexaSpin(200,math.radians(t)),math.radians(t)) for t in range(0,360,2)]

    def modeDeactivated(self): # When deactivated, re-impose the graphics-file canvas
        self.canvas.pack_forget()
        self.app._canvas.pack()    

    def polarize(self,x,y,r,t): # Generates a coordinate based on polar instrucitons
        dX = r * math.cos(t)
        dY = r * math.sin(t)
        return (x+dX,y+dY)

    def hexaSpin(self,r,t): # Generates points on a hexagon from polar instructions
        n = 6 # n controls number of sides
        return r * math.cos(math.pi/n) / (math.cos(t % (math.pi/(n/2)) - math.pi/n))
        
    def keyPressed(self,event): # Switch between modes (temporary)
        self.app.setActiveMode(self.app.sandboxMode)

    def timerFired(self): # Increment the animation
        self.spin += 5
        self.squares.append(self.squares.pop(0))

    def drawSquare(self,i,square,canvas): # Drawing each square based on polar formula
        x0, y0 = square
        s = 70
        x1, y1 = x0 + s * math.cos(math.radians(i)), y0 + s * math.sin(math.radians(i))
        x2, y2 = x0 + s * math.cos(math.radians(i+90)), y0 + s * math.sin(math.radians(i+90))
        x3, y3 = x0 + s * math.cos(math.radians(i+180)), y0 + s * math.sin(math.radians(i+180))
        x4, y4 = x0 + s * math.cos(math.radians(i-90)), y0 + s * math.sin(math.radians(i-90))
        red = int(85 * math.sin(2*math.pi*(i/360)) + 170)
        green = int(85 * math.sin(2*math.pi*((i+120)/360)) + 170)
        blue = int(85 * math.sin(2*math.pi*((i+240)/360)) + 170)
        color = '#' + hex(red)[2:] + hex(green)[2:] + hex(blue)[2:]
        canvas.create_polygon(x1,y1,x2,y2,x3,y3,x4,y4,fill=color,width=0)

    def redrawAll(self,canvas): # Clear canvas with each tick & draw all squares
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(0,0,self.width,self.height,fill='Black')
        for i in range(6):
            sX, sY = self.squares[int(i*30+self.spin/5)%180]
            self.canvas.create_text((sX+800)/3,(sY+500)/3,font='Times 30',fill='White',text='KIMCHI'[i])
        for i,square in enumerate(self.squares):
            self.drawSquare(9*i+self.spin%360,square,self.canvas)
        self.canvas.create_text(800-(sX+2000)/6,490+(sY+1250)/6,font=f'Times {int(2*math.sin(self.spin/5)+12)}',fill='White',text='press any key to begin')

class TopLevel(ModalApp): # Outermost app class
    
    globalCanvas = None
    curMode = 0
    width = height = 800

    def appStarted(self):
        self._root.resizable(False, False)
        TopLevel.Vars = [None for i in range(maxVars)]
        self.generateAtoms(25)
        self.generateIO(25)
        self.generateFunctions(25)
        TopLevel.Defaults = [blockLibrary['Null'] for i in range(maxVars)]
        TopLevel.VariableSlots = VarSetFunction('Set Variables')
        TopLevel.FuncSlots = GenesisFunction('Create Function')
        self.splashMode = SplashMode()
        self.creationMode = CreationMode()
        self.sandboxMode = Sandbox()
        self.setActiveMode(self.splashMode)
        self.funcStrings = ''

    @staticmethod
    def globalException(e): # What to put on screen instead of an exception that crashes TKinter
        x0, y0 = TopLevel.width * 0.825, TopLevel.height * 0.8625
        TopLevel.globalCanvas.create_oval(x0,y0,x0+20,y0+20,fill='#f2e422',width=0)
        TopLevel.globalCanvas.create_text(x0,y0+25,text=chunkifyString(customError(e),25),anchor='nw',font='Times 12',fill='White')

    def generateAtoms(self,atomX): # Generates all Atoms
        out = [VarAtom(TopLevel.Vars[0],atomX,15,True,'V0')]
        out += [VarAtom(TopLevel.Vars[i],atomX,out[-1].y+(Dragger.targetSize*2*i),True,f'V{i}') for i in range(1,maxVars)]
        out += [Atom(i,atomX,out[-1].y+(Dragger.targetSize*2*(i+1)),True,str(i)) for i in range(11)]
        out += [Atom(-1,atomX,out[-1].y+Dragger.targetSize*2,True,'-1')]
        out += [Atom(round(math.tau,8),atomX,out[-1].y+Dragger.targetSize*2,True,chr(428))]
        out += [Atom(round(math.e,8),atomX,out[-1].y+Dragger.targetSize*2,True,'e')]
        out += [Atom(float('inf'),atomX,out[-1].y+Dragger.targetSize*2,True,'∞')]
        out += [Atom(float('-inf'),atomX,out[-1].y+Dragger.targetSize*2,True,'-∞')]
        out += [Atom(chr(i+65),atomX,out[-1].y+(Dragger.targetSize*2*(i+1)),True,chr(i+65)) for i in range(26)]
        out += [Atom(' ',atomX,out[-1].y+Dragger.targetSize*2,True,'space')]
        out += [Atom("''",atomX,out[-1].y+Dragger.targetSize*2,True,'Silent')]
        out += [Atom(Axis.Axis(),atomX,out[-1].y+Dragger.targetSize*2,True,'Empty')]
        out += [Atom(True,atomX,out[-1].y+Dragger.targetSize*2,True,'True')]
        out += [Atom(False,atomX,out[-1].y+Dragger.targetSize*2,True,'False')]
        out += [Atom(None,atomX,out[-1].y+Dragger.targetSize*2,True,'Null')]
        out += [Atom(None,atomX,out[-1].y+Dragger.targetSize*2,True,chr(955))]
        out += [Atom(None,atomX,out[-1].y+Dragger.targetSize*2,True,chr(956))]

    def generateFunctions(self,atomX): # Generates all pre-built functions from a .txt file
        funcStr = open('PreFunctions.txt','r')
        iFunctions = funcStr.read().splitlines()
        funcStr.close()
        out = [Function.genesis(iFunctions[0],atomX,Dragger.targetSize*2*(len(blockLibrary)+0.75))]
        for i in iFunctions[1:]:
            out += [Function.genesis(i,atomX,out[-1].y+Dragger.targetSize*2)]

    def generateIO(self,atomX): # Generates all Atoms
        out = [IOAtom(None,atomX,15,True,'I0')]
        out += [IOAtom(None,atomX,out[-1].y+(Dragger.targetSize*2*i),True,'I'+str(i)) for i in range(1,10)]

TopLevel(width=800,height=800)
