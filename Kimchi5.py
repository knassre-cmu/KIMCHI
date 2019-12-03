import copy, math, random, string, functools, os
from tkinter import *
from tkinter import simpledialog
import Axis
from cmu_112_graphics import * # From https://www.cs.cmu.edu/~112/notes/notes-animations-part1.html

# To do list:
# fix the evaluator section in Function Creator mode
# improve capabilities of random
# widen screen
# add custom backgrounds (w saving?)
# fix Quandry & Mystery in tutorial mode (delete upon exiting)

blockLibrary = {} # Dictionary of all known functions/atoms (used for defaults & drawing function library)
customFunctions = set() # Set of all custom functions that will be exported with saving
ioLibrary = {} # Dictionary of input atoms used for function genesis
maxVars = 16 # Number of pink variables used in sandbox mode

spiralCords = [] # Cordinates used to draw a spiral
for t in range(25):
    spiralCords.append(605+t*math.cos(t)/1.3)
    spiralCords.append(625+t*math.sin(t)/1.3)
    
def popupBox(question): # My own implementation of getUserInput
    return simpledialog.askstring('', question)

def getKimchiFiles(): # Searches through the current directory tto find Kimchi files
    return list(map(lambda y: y[15:][:-4], filter(lambda x: x[:14] == '<<KimchiFile>>' and x[-4:] == '.txt',os.listdir(os.getcwd()))))

def fillBlockWithInputs(f,i): # Parses through list of strings representing a function to add inputs
    if f in list(ioLibrary.keys()): return i[int(f[1])]
    if isinstance(f,str): return f
    return list(map(lambda x: fillBlockWithInputs(x,i),f))

def callBlockWithInputs(f,parent): # Parses through a list of strings representing a function (with inputs) to call it
    if isinstance(f,str) and f in blockLibrary: return blockLibrary[f]() # If at an Atom, just return it
    if not isinstance(f,list): return f # If at a value, just return it
    func = blockLibrary[f[0]]
    try:
        if isinstance(func,ControlFunction): # If a control function, implement HOF (reduceLambda knows how to parse through strings)
            return func.value(f[1],callBlockWithInputs(f[2],parent))
        if isinstance(func,CustomFunction): # If a custom function, retrieve it's data string
            if CreationMode.frankenstein != None and f[0] == CreationMode.frankenstein.name: # If calling the function currently being worked on, get it from the class attributtes
                data = TopLevel.FuncSlots.slots[0].FuncToString()
            else: data = func.dataString
            return callBlockWithInputs(fillBlockWithInputs(data,list(map(lambda x: callBlockWithInputs(x,parent), f[1:]))),parentt)
        if func.name == 'if else': # Sequential-evaluation logic (if-else, switch) requires manual implementation
            if callBlockWithInputs(f[1],parent): return callBlockWithInputs(f[2],parent)
            return callBlockWithInputs(f[3],parent)
        if func.name == 'switch':
            for i in range(1,len(f[1])):
                if callBlockWithInputs(f[1][i],parent): return callBlockWithInputs(f[2][i],parent)
            return blockLibrary['Null']
        pRand = (parent.rid + (hash(str(f)) % 1000)/1000) % 1
        if f[0] == 'random int':
            lo = callBlockWithInputs(f[1],parent)
            hi = callBlockWithInputs(f[2],parent)
            return int((pRand* (hi-lo)) + lo)
        if f[0] == 'random float':
            lo = callBlockWithInputs(f[1],parent)
            hi = callBlockWithInputs(f[2],parent)
            return (pRand * (hi-lo)) + lo
        elif f[0] == 'random choice':
            l = callBlockWithInputs(f[1],parent)
            return l[int(pRand*len(l))]
        return func.value(*map(lambda x: callBlockWithInputs(x,parent), f[1:]))
    except Exception as e: # Catch all errors under the global exception
        TopLevel.globalException(e)
        try: return func()
        except: return None

def customError(e):
    return str(e)

def darkenColor(h,level=1): # Takes in a hex color and returns a darkened version
    conversion = {'f':'d','e':'c','d':'b','c':'a','b':'9','a':'8','9':'7',
    '8':'6','7':'5','6':'4','5':'3','4':'2','3':'1','2':'0','1':'0','0':'0'}
    out = '#'
    for i in h[1:]: out += conversion[i]
    if level <= 1: return out
    return darkenColor(out,level-1)

def chunkifyString(s,l): # Chops up a string into lines of length l
    out, row = [], ''
    for w in s.split():
        if len(w) + len(row) > l:
            out.append(row)
            row = w
        else: row += ' ' + w
    return '\n'.join(out+[row]).strip()

def rgb2hex(r,g,b): # Returns a hexedecimal color string based on 3 rgb integers
    h1, h2, h3 = hex(r)[2:], hex(g)[2:], hex(b)[2:]
    if r < 16: h1 = '0' + h1
    if g < 16: h2 = '0' + h2
    if b < 16: h3 = '0' + h3
    return '#' + h1 + h2 + h3

def ifThenElse(condition,option1,option2): # If-else logic (cased for different input types)
    if isinstance(condition,bool): b = condition
    else: b = condition()
    if isinstance(option1,Dragger): I = option1()
    else: I = option1
    if isinstance(option2,Dragger): E = option2()
    else: E = option2
    if b: return I
    return E

def switch(cases,outcomes): # Switch statements (to simplify excessive if-else nesting)
    for i in range(len(cases())):
        if cases()[i]: return outcomes()[i]
    return blockLibrary['Null']

def minimize(f,l): # Loop over a list and pick out the item with the smallest output to some function
    maxInput = None
    maxOutput = float('inf')
    for n,i in enumerate(l):
        hof = replaceLambda(f,i,0,n) # Parses through HOF and replace lambda Atoms with current item
        if hof < maxOutput:
            maxInput = l[i]
            maxOutput = hof
    return maxInput

def optimize(f,l): # Loop over a list and pick out the item with the largest output to some function
    maxInput = None
    maxOutput = float('-inf')
    for n,i in enumerate(l):
        hof = replaceLambda(f,i,0,n) # Parses through HOF and replace lambda Atoms with current item
        if hof > maxOutput:
            maxInput = l[i]
            maxOutput = hof
    return maxInput

def findFirst(f,l): # Loop over a list and pick out the first item that returns true for some function
    for n,i in enumerate(l):
        if replaceLambda(f,i,0,n): return l # Parses through HOF and replace lambda Atoms with current item
    return None

def get(t,d,default): # Treat a list of 2-item lists as a dicitonary with the .get() method
    for i in d:
        if i[0] == t: return i[1]
    return default

def kSort(f,l): # Sorting HOF (insertion sort)
    out = []
    for n,i in enumerate(l):
        j, k = len(out), replaceLambda(f,i,0,n) # Parses through HOF and replace lambda Atoms with current item
        while j > 0:
            if k < replaceLambda(f,out[j-1],0,n): j -= 1
            else: break
        out.insert(j,i)
    return Axis.Axis(*out)

def kMap(f,l): # Mapping HOF
    out = []
    for n,i in enumerate(l): out.append(replaceLambda(f,i,0,n)) # Parses through HOF and replace lambda Atoms with current item
    return Axis.Axis(*out)

def kFilter(f,l): # Filtering HOF
    out = []
    for n,i in enumerate(l):
        if replaceLambda(f,i,0,n): out.append(i) # Parses through HOF and replace lambda Atoms with current item
    return Axis.Axis(*out)

def kCombine(f,l): # Combination HOF
    out = l[0]
    for n,i in enumerate(l[1:]): out = replaceLambda(f,out,i,n) # Parses through HOF and replace lambda & mew Atoms with current items
    return out

def replaceLambda(f,i,m=None,k=0): # Replaces all lambda atoms with the desired input (modified to work with string parser)
    if f == chr(955) or (isinstance(f,Atom) and f.name == chr(955)): return i # Base case: replace lambda with input
    elif f == chr(956) or (isinstance(f,Atom) and f.name == chr(956)): return m # Base case: replace mew with input
    elif isinstance(f,CustomFunction): # What to do if the HOF is a custom function
        return callBlockWithInputs(fillBlockWithInputs(f.dataString,list(map(lambda x: replaceLambda(x,i,m,k),f.slots))),f)
    elif isinstance(f,Function): # What to do if the HOF is a normal function
        if f.name in ['random int','random float','random choice']:
            pRand = (f.rid * (k+3)**2) % 1
            if f.name == 'random int':
                lo = f.operands[0]()
                hi = f.operands[1]()
                return int((pRand * (hi-lo)) + lo)
            if f.name == 'random float':
                lo = f.operands[0]()
                hi = f.operands[1]()
                return (pRand * (hi-lo)) + lo
            elif f.name == 'random choice':
                l = f.operands[0]()
                return l[int(pRand*len(l))]
        return f.value(*map(lambda x: replaceLambda(x,i,m,k),f.operands))
    elif isinstance(f,list): # What to do if the HOF a datastring Function
        return blockLibrary[f[0]].value(*map(lambda x: replaceLambda(x,i,m,k),f[1:]))
    elif isinstance(f,str): # What to do if the HOF is a datastring Atom
        return blockLibrary[f].value
    return f() # Backup case: just let __call__ sort it out

def randomList(n,k): # Return a list of n random floats with seed k
    out = [round((hash(k/math.pi)/100000)%1,8)] # use hashing to generate pseudo randomness
    for i in range(n-1):
        out.append(round((out[-1]+hash(out[-1]/math.pi)/100000)%1,8))
    return Axis.Axis(*out)

class Dragger(object): # Class of all draggable blocks

    cloneList = [] # List of all block clones (in sandbox mode)
    cloneList2 = [] # List of all block clones (in function mode)
    fontSize = 17
    targetSize = 10

    def exportBlock(self):
        bString = [self.name,self.x,self.y]
        if isinstance(self,Function): bString += [i.exportBlock() if i != None else None for i in self.slots]
        return bString

    @staticmethod
    def exportClones():
        out = ''
        for i in [TopLevel.VariableSlots] + Dragger.cloneList:
            if i.name == 'Create Function' or (i.name == 'Set Variables' and 'Set Variables' in out): continue
            if i.parent == None: out += 'v: ' + str(i.exportBlock()) + '\n'
        return out.strip()
    
    @staticmethod
    def superGenesis(bString,parent=None):
        if bString == None: return None
        b2d = blockLibrary[bString[0]]
        x, y = bString[1], bString[2]
        value, name = b2d.value, b2d.name
        if isinstance(b2d,Atom):
            if isinstance(b2d,VarAtom): return VarAtom(value,x,y,False,name,parent)
            else: return Atom(value,x,y,False,name,parent)
        else:
            defaults = b2d.defaults
            slots = [None for i in range(len(b2d.defaults))]
            if isinstance(b2d,CustomFunction): f = CustomFunction(name,value,x,y,False,defaults,slots,parent,b2d.color,b2d.dataString)
            else: f = eval(b2d.fType)(name,value,x,y,False,defaults,slots,parent)
            f.slots = [Dragger.superGenesis(i,f) for i in bString[3:]]
            f.getDimensions()
            f.slotify()
            f.getOperands()
            return f
    
    def __bool__(self): # If treated as a boolean, return boolean of value
        return bool(self())

    def touching(self,x,y,app,awaken=False): # Is this block being touched
        if self.cloneable: # If cloneable, create a clone (and adjust for scrolling)
            if (x-self.x)**2 + (y+app.scrollY-self.y)**2 < 100:
                if isinstance(self,Atom):
                    if type(self) == VarAtom: return VarAtom(self.value,x,y+app.scrollY2,False,self.name)
                    elif type(self) == IOAtom:
                        if int(self.name[1]) >= GenesisFunction.using: return
                        return IOAtom(self.value,x,y+app.scrollY2,False,self.name)
                    return Atom(self.value,x,y+app.scrollY2,False,self.name)
                if type(self) == CustomFunction:
                    return CustomFunction(self.name,self.value,x,y,False,self.defaults,[None for i in range(len(self.operands))],None,self.color,self.dataString)
                return eval(self.fType)(self.name,self.value,x,y,False,self.defaults,[None for i in range(len(self.operands))],None)
        elif (x-self.x)**2 + (y+app.scrollY2-self.y)**2 < 100 or awaken: # Else, just move it (and bring to top)
            if self.name != 'Set Variables':
                if type(app) == Sandbox:
                    Dragger.cloneList.remove(self)
                    Dragger.cloneList.append(self)
                elif type(app) == CreationMode:
                    Dragger.cloneList2.remove(self)
                    Dragger.cloneList2.append(self)
            return self

    def __eq__(self,other): # Compares Draggers
        return type(self) == type(other) and self.name == other.name and self.x == other.x and self.y == other.y and self.value == other.value

    def FuncToString(self): # Converts Draggers into strings that can be read by custom functions
        if isinstance(self,Atom):
            return self.name
        return [self.name] + list(map(lambda x: x.FuncToString(),self.operands))

    def exception(self): # Turns items into yellow circles if causing an error
        try:
            self.value(*map(lambda x: x(), self.operands))
            return False
        except:
            return True

    def superSlotify(self): # Reset the slotting of items during a load
        for i in Dragger.cloneList + [TopLevel.VariableSlots]:
            if isinstance(i,Function):
                if i.name == 'Create Function':
                    continue
                slot = i.insideSlot(self.x,self.y)
                if slot != None:
                    i.slots[slot] = self
                    self.parent = i
                    i.slotify()
                    if i.name == 'Set Variables':
                        TopLevel.Vars[slot] = self
                        blockLibrary[f'V{slot}'].value = self.value
                        for i in Dragger.cloneList:
                            for j in range(maxVars):
                                if i.name == 'V'+str(j):
                                    try: i.value = TopLevel.Vars[j]
                                    except: pass

class Function(Dragger): # Class of all functions
    
    def __init__(self,fType,name,value,x,y,cloneable,defaults,slots,parent=None):
        self.rid = random.random() # Random numbers cached for each random function
        self.fType, self.name, self.value, self.x, self.y = fType, name, value, x, y
        self.cloneable, self.defaults, self.slots = cloneable, defaults, slots
        self.parent = parent
        self.getDimensions()
        self.operands = slots[:]
        self.getOperands()
        self.buffer = 5
        if self.cloneable: blockLibrary[self.name] = self
        elif TopLevel.curMode == 2: Dragger.cloneList2.append(self)
        elif TopLevel.curMode == 1: Dragger.cloneList.append(self)

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

    def __repr__(self): # How to print for debugging:
        return f'{self.name} ({", ".join(map(lambda x: str(x),self.operands))}) => {str(self())}'

    def drawLibraryFunction(self,canvas,app): # Draws the cloneable functions in the library
        clickColor = '#b91613' # Red circle, since not being held
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = 150, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x,y0-app.scrollY,x1,y1-app.scrollY,fill=self.color,width=0)
        canvas.create_oval(x0,y0-app.scrollY,x2,y1-app.scrollY,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY,anchor='w',text=self.name,font=f'Times {Dragger.fontSize}')

    def drawFunction(self,canvas,app): # Draws the cloned functions
        if self == app.holding:
            try: clickColor = self.clickColor
            except: clickColor = '#2e9121'
        else: clickColor = '#b91613' # Red circle if not being held
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

    def __call__(self):
        try:
            if self.name == 'random int':
                lo = self.operands[0]()
                hi = self.operands[1]()
                return int((self.rid * (hi-lo)) + lo)
            if self.name == 'random float':
                lo = self.operands[0]()
                hi = self.operands[1]()
                return (self.rid * (hi-lo)) + lo
            elif self.name == 'random choice':
                l = self.operands[0]()
                return l[int(self.rid*len(l))]
            else:
                val = self.value(*map(lambda x: x(), self.operands))
                if isinstance(val,float):
                    return round(val,8)
                return val
        except Exception as e: # If inputs are invalid, return the defualt and print an error
            TopLevel.globalException(e)
            return self.value(*map(lambda x: x(), self.defaults))

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

    def __call__(self): # If being treated like a function, call the lambda expression
        try:
            if self.name == 'if else' or self.name == 'switch':
                val = self.value(*self.operands)
            else:
                val = self.value(*map(lambda x: x(), self.operands))
            if isinstance(val,float):
                return round(val,8)
            return val
        except Exception as e: # If inputs are invalid, return the defualt and print an error
            print(e)
            TopLevel.globalException(e)
            return self.value(*map(lambda x: x(), self.defaults))

class IterableFunction(Function):
    color = '#ea5639'
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
    def __init__(self,name,vals):
        super().__init__('VarSetFunction',name,None,165,20,False,TopLevel.Defaults,vals,None)
        self.color = '#e85f64'
        self.buffer = 35

    def wipeVariables(self): # Wipes the variable slots clean during a reset
        TopLevel.Vars = [None for i in range(maxVars)]
        self.slots = [None for i in range(maxVars)]

    def __call__(self): # If being called, do nothing
        pass

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
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None,color=None,dataString=''):
        super().__init__('CustomFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = color
        self.dataString = dataString
        if cloneable:
            customFunctions.add(self)

    def __call__(self):
        try:
            if CreationMode.frankenstein != None and self.name == CreationMode.frankenstein.name:
                me = TopLevel.FuncSlots.slots[0].FuncToString()
            else:
                me = blockLibrary[self.name].dataString
            val = callBlockWithInputs(fillBlockWithInputs(me,list(map(lambda x: x(), self.slots))),self)
            return val
        except Exception as e:
            TopLevel.globalException(e)
            return None

    def __hash__(self):
        return hash(self.name)

    def exportFunction(self):
        return (self.name,self.color,len(self.slots),self.dataString)

    @staticmethod
    def uberGenesis(s):
        n, c, i, f = s
        new = CustomFunction(n,None,25,sorted(list(blockLibrary.values()),key=lambda x: x.y)[-1].y+2*Dragger.targetSize,
        True,[blockLibrary['Null'] for i in range(i)],[None for j in range(i)],None,c,f)

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
        elif TopLevel.curMode == 1: Dragger.cloneList.append(self)
        
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

    def __init__(self,value,x,y,cloneable,name,parent=None):
        super().__init__(value,x,y,cloneable,name,parent)
        self.xMin = self.xMax = Dragger.targetSize + 10 + len(f'Input {int(self.name[1])+1}') * Dragger.fontSize/2.3

    def drawLibraryAtom(self,canvas,app): # Draws the cloneable atoms in the library
        clickColor = '#b91613' # Red circle since not currently being held
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = 150, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x,y0-app.scrollY,x1,y1-app.scrollY,fill=self.color,width=0)
        canvas.create_oval(x0,y0-app.scrollY,x2,y1-app.scrollY,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY,anchor='w',text=f'Input {int(self.name[1])+1}',font=f'Times {Dragger.fontSize}')

    def drawAtom(self,canvas,app): # Draws the cloned atoms
        if self == app.holding: clickColor = '#2e9121' # Green circle if currently being held
        else: clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + Dragger.targetSize + self.xMin, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x+2,y0-app.scrollY2+2,x1-2,y1-app.scrollY2-2,fill=self.color,width=2,outline=darkenColor(self.color))
        canvas.create_oval(x0,y0-app.scrollY2,x2,y1-app.scrollY2,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY2,anchor='w',text=f'Input {int(self.name[1])+1}',font=f'Times {Dragger.fontSize}')

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
        self.background = 0
        self.backgroundMax = 10

    def modeActivated(self): # Scrap the graphics file canvas so that I can handle exceptions myself
        self.scrollMax = sorted(list(blockLibrary.values()),key=lambda x: x.y)[-1].y+2*Dragger.targetSize - 800
        self.app._canvas.pack_forget()
        self.canvas = Canvas(self.app._root,width=800,height=800)
        self.canvas.pack()
        TopLevel.globalCanvas = self.canvas
        TopLevel.curMode = 1
        for i in Dragger.cloneList: i.superSlotify() # Prevents functions from "sticking" when loaded

    def modeDeactivated(self): # When deactivated, re-impose the graphics-file canvas
        self.canvas.pack_forget()
        self.app._canvas.pack()

    def pressingButtons(self,x,y):
        options = [self.app.splashMode,self.app.splashMode,self.app.splashMode]
        yVals = [575, 625, 675, 725, 775]
        for i in range(5):
            if (x > 590 or ((x-590)**2 + (y-yVals[i])**2) < 625) and x < 630 and y < yVals[i]+25 and y > yVals[i]-25:
                if i == 0: TopLevel.newSave()
                elif i == 1: self.background += 1; self.background %= self.backgroundMax
                elif i == 2: self.getFunctionInfo()
                elif i == 3: self.app.setActiveMode(self.app.splashMode)
                elif i == 4: 
                    if popupBox('Type DELETE to clear the sandbox') == 'DELETE':
                        TopLevel.wipeBlocks()
                        TopLevel.VariableSlots.getDimensions()

    def getFunctionInfo(self):
        if not CreationMode.done:
            self.app.setActiveMode(self.app.creationMode)
        else:
            fName = popupBox('What is the function\'s name?')
            if fName == None: pass
            elif fName == '' or len(fName) > 14 or fName in map(lambda x: x.name, blockLibrary.values()):
                nextStep = popupBox(f'"{fName}" is not a valid name.\nPress "ok" to try again.')
                if nextStep != None: self.getFunctionInfo()
            else:
                fType = popupBox(f'What type of function is {fName}:\nArithmetic, Letter, Iterable or Predicate?')
                if fType == None: pass
                elif fType == None or fType.lower() not in ['letter','arithmetic','iterable','predicate']:
                    if fType == 'control':
                        nextStep = popupBox(f'You cannot make control functions.\nPress "ok" to try agian.')
                    else:
                        nextStep = popupBox(f'"{fType}" is not a valid function type.\nValid types: arithmetic, predicate, letter, iterable\nPress "ok" to try agian.')
                    if nextStep != None: self.getFunctionInfo()
                else:
                    fInpt = popupBox(f'How many inputs does {fName} have?')
                    if fInpt == None: pass
                    elif fInpt not in map(lambda x: str(x), range(1,10)):
                        nextStep = popupBox(f'"{fInpt}" is not a valid number of inputs.\nPress "ok" to try agian.')
                        if nextStep != None: self.getFunctionInfo()
                    else:
                        self.app.funcStrings = (fName,fType,fInpt)
                        GenesisFunction.curName = fName
                        GenesisFunction.color = eval(f'{fType.lower().title()}Function.color')
                        GenesisFunction.using = int(fInpt)
                        for block in ioLibrary.values(): block.getColor()
                        CreationMode.done = False
                        CreationMode.megaGenesis(*self.app.funcStrings)
                        TopLevel.FuncSlots.slots = [None]
                        TopLevel.FuncSlots.getDimensions()
                        Dragger.cloneList2 = []
                        self.app.setActiveMode(self.app.creationMode)

    def mousePressed(self,event): # Check for scrolling & dragging Atoms/Functions
        self.pressingButtons(event.x,event.y)
        if event.x < 10: self.scrolling = True
        elif event.x < 640 and event.x > 630: self.scrolling2 = True
        elif event.x > 790: self.scrolling3 = True
        else:
            for i in list(blockLibrary.values()) + Dragger.cloneList:
                if i == None or i.name == 'Create Function': continue
                contact = i.touching(event.x,event.y,self)
                if contact != None:
                    if contact.name == 'Set Variables': return
                    self.holding = contact
                    self.holding.rid = random.random()
                    contact.clickColor = '#2e9121' # Green circle if currently being held (no exception handling)
                    if self.holding.parent != None:
                        for i in range(len(self.holding.parent.slots)):
                            if self.holding.parent.slots[i] == self.holding:
                                if self.holding.parent.name == 'Set Variables':
                                    TopLevel.Vars[i] = None
                                    blockLibrary[f'V{i}'].value = None
                                    for k in Dragger.cloneList:
                                        for j in range(maxVars):
                                            if k.name == 'V'+str(j):
                                                try: k.value = TopLevel.Vars[j]
                                                except: pass
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
        if self.holding == [None] or self.holding.name == 'Set Variables': return
        self.holding.x = event.x
        self.holding.y = event.y + self.scrollY2
        if isinstance(self.holding,Function):
            self.holding.slotify()

    def mouseReleased(self,event): # Stop scrolling & release dragged Atoms/Funcitons
        self.scrolling = False
        self.scrolling2 = False
        self.scrolling3 = False
        if self.holding == [None] or self.holding.name == 'Set Variables': self.holding = [None]; return
        self.holding.x = event.x
        self.holding.y = event.y + self.scrollY2
        if isinstance(self.holding,Function):
            self.holding.slotify()
        xEdge = event.x + self.holding.xMax/2
        if event.x - Dragger.targetSize < 150 or xEdge > self.width*0.8:
            Dragger.cloneList.remove(self.holding)
            if isinstance(self.holding,Function): self.holding.killChildren()
        else:
            for i in Dragger.cloneList + [TopLevel.VariableSlots]:
                if isinstance(i,Function):
                    if i.name == 'Create Function': continue
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
                                        try: i.value = TopLevel.Vars[j]
                                        except: pass
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
            if v == None or v() == None: out.append(f'V{i} = Null')
            else:
                if isinstance(v(),float): out.append(chunkifyString(f'V{i} = {round(v(),8)}',28))
                else: out.append(chunkifyString(f'V{i} = {v()}',28))
        Sandbox.VarString = '\n'.join(out)
        canvas.create_text(x0,y0,anchor='nw',text=Sandbox.VarString,fill='Green',font='Times 10')

    def drawButtons(self,canvas):
        r, x0, x1 = 25, 590, 630
        y4, y0, y1, y2, y3, y5 = 550, 600, 650, 700, 750, 800
        xA, yA = (x0+x1-12)/2, y1 + 6
        xB, yB = x0 - 6, (y1 + y2)/2
        xC, yC = x1 - 6, y2 - 6
        c1A, c2A, c3A, c4A, c5A = '#ffee99', '#aaffaa', '#99bbff', '#ff99bb', '#999999'
        c1B, c2B, c3B, c4B, c5B = darkenColor(c1A,2), darkenColor(c2A,2), darkenColor(c3A,2), darkenColor(c4A,2), darkenColor(c5A,2)
        if type(self) == Sandbox:
            canvas.create_rectangle(x0,y4,x1,y0,width=0,fill=c4B) # Draw save button
            canvas.create_oval(x0-r/1.5,y4,x0+r/1.5,y0,width=0,fill=c4B)
            canvas.create_text((x0+x1-10)/2,(y0+y4)/2,fill=c4A,text='S',font='Times 40 bold')
            canvas.create_rectangle(x0,y3,x1,y5,width=0,fill=c5B) # Draw trash button
            canvas.create_oval(x0-r/1.5,y3,x0+r/1.5,y5,width=0,fill=c5B)
            canvas.create_polygon(x0,y3+20,x0+2.5,y5-10,x1-17.5,y5-10,x1-15,y3+20,xA+6,y3+10,xA-8,y3+10,width=0,fill=c5A)
            canvas.create_line(x0+7,y5-10,x0+5,y3+20,fill=c5B,width=2)
            canvas.create_line(x1-21.5,y5-10,x1-19.5,y3+20,fill=c5B,width=2)
            canvas.create_line(xA-1,y5-10,xA-1,y3+20,fill=c5B,width=2)
            canvas.create_line(x0,y3+20,x1-15,y3+20,fill=c5B,width=2)
        canvas.create_rectangle(x0,y0,x1,y1,width=0,fill=c1B) # Draw animation-engine button
        canvas.create_oval(x0-r/1.5,y0,x0+r/1.5,y1,width=0,fill=c1B)
        canvas.create_rectangle(x0-2,y0+9,x1-9,y1-9,width=4,outline=c1A)
        canvas.create_line(*spiralCords,width=2,fill=c1A,smooth=True)
        canvas.create_rectangle(x0,y1,x1,y2,width=0,fill=c2B) # Draw function-cretor button
        canvas.create_oval(x0-r/1.5,y1,x0+r/1.5,y2,width=0,fill=c2B)
        canvas.create_polygon(xB,yB-5,xB,yB,xB,yB+5,xA-5,yB+5,xA-5,yC,xA,yC,xA+5,yC,xA+5,yB+5,xC,yB+5,
        xC,yB,xC,yB-5,xA+5,yB-5,xA+5,yA,xA,yA,xA-5,yA,xA-5,yB-5,width=0,fill=c2A,smooth=True)
        canvas.create_rectangle(x0,y2,x1,y3,width=0,fill=c3B) # Draw home button
        canvas.create_oval(x0-r/1.5,y2,x0+r/1.5,y3,width=0,fill=c3B)
        canvas.create_polygon(x0,y2+20,x0,y3-10,x1-15,y3-10,x1-15,y2+20,xA-1,y2+10,width=0,fill=c3A)

    def drawLibraryScroll(self,canvas): # Draw the scrollbar(s)
        y2 = min(max(10,self.height * self.scrollY / self.scrollMax),790)
        y3 = min(max(5,self.height * self.scrollY2 / self.scrollMax2),790)
        canvas.create_rectangle(0,0,15,self.height,fill='#d1d1d2',width=0)
        canvas.create_oval(4,y2-7,14,y2+7,fill='#a8a7a9',width=0)
        canvas.create_rectangle(630,0,640,self.height,fill='#d1d1d2',width=0)
        canvas.create_oval(632,y3-3,638,y3+3,fill='#a8a7a9',width=0)

    def drawTartan(self,canvas,*stripes):
        for x0,y0,x1,y1,c,w in stripes:
            if y0 < y1: xA, yA, xB, yB, xC, yC, xD, yD = x0+w, y0-w, x1+w, y1-w, x1-w, y1+w, x0-w, y0+w
            else: xA, yA, xB, yB, xC, yC, xD, yD = x0+w, y0+w, x1+w, y1+w, x1-w, y1-w, x0-w, y0-w
            canvas.create_polygon(xA,yA,xB,yB,xC,yC,xD,yD,fill=c,width=0)

    def drawBackground(self,canvas):
        if self.background < 2: canvas.create_rectangle(150,0,0.8*self.width,self.height,fill='Black',width=0)
        if self.background == 1: 
            dT = 2.95
            hexaColors = ['#ff99aa','#ffbb99','#ffff99','#99ffaa','#99aaff','#ffaaff']
            for i in range(153): canvas.create_line(400+2*i*math.cos(math.pi*i/dT),400+2*i*math.sin(math.pi*i/dT),400+2*i*math.cos(math.pi*(i+1)/dT),400+2*i*math.sin(math.pi*(i+1)/dT),width=3,fill=hexaColors[i%6])
        elif self.background == 2: 
            for x,y in self.app.vBars: canvas.create_line(x,y-10,x,y+10,width=3)
            for x,y in self.app.hBars: canvas.create_line(x-10,y,x+10,y,width=3)
        elif self.background == 3: canvas.create_line(*functools.reduce(lambda x,y: x+y,self.app.dragonCurve),width=2)
        elif self.background == 4: self.drawTartan(canvas,(200,-100,700,400,'#ffff99',7),(0,0,700,700,'#99aaff',10),(0,900,700,200,'#99aaff',7),(0,800,700,100,'#ffff99',2),(0,775,700,75,'#ffff99',2),(400,-50,700,250,'#99aaff',20),(0,600,700,-100,'#ffff99',17),(0,530,700,-180,'#ffff99',4),(0,380,700,1080,'#99aaff',2))
        elif self.background == 5: 
            for i in self.app.polka: self.canvas.create_oval(i[0]-i[2],i[1]-i[2],i[0]+i[2],i[1]+i[2],fill=i[3],width=0)
        elif self.background == 6: 
            for i in self.app.diamonds: self.canvas.create_polygon(i[0]-i[2],i[1],i[0],i[1]+i[2],i[0]+i[2],i[1],i[0],i[1]+i[2]/2,fill=i[3],width=0)
        elif self.background == 7:
            for i in self.app.isoms: self.drawIsometric(canvas,i[0],i[1])
        elif self.background == 8:
            for i in self.app.isoms: self.drawIsometric2(canvas,i[0],i[1])
        elif self.background == 9: self.canvas.create_image(400,400,image=self.app.kLogo)
        canvas.create_rectangle(0.8*self.width,0,self.width,self.height,fill='Black',width=0)

    def drawIsometric(self,canvas,x2,y2):
        x0, x1, x3, x4 = x2-40, x2-20, x2+20, x2+40
        y0, y1, y3, y4, y5, y6 = y2 - 15, y2-10, y2 + 5, y2+10, y2+25, y2+30
        canvas.create_polygon(x0,y1,x1,y0,x2,y1,x3,y0,x4,y1,x2,y2,fill=darkenColor('#ff99aa',2),width=0)
        canvas.create_polygon(x0,y1,x0,y3,x1,y4,x1,y5,x2,y6,x2,y2,fill=darkenColor('#ffff99',0),width=0)
        canvas.create_polygon(x4,y1,x4,y3,x3,y4,x3,y5,x2,y6,x2,y2,fill=darkenColor('#99aaff',2),width=0)

    def drawIsometric2(self,canvas,x2,y2):
        x0, x1, x3, x4 = x2-38, x2-18, x2+18, x2+38
        y6, y5, y4, y3, y1, y0 = y2 - 15, y2-10, y2 + 5, y2+10, y2+25, y2+30
        canvas.create_polygon(x0,y1,x1,y0,x2,y1,x3,y0,x4,y1,x2,y2,fill=darkenColor('#99aaff',2),width=0)
        canvas.create_polygon(x0,y1,x0,y3,x1,y4,x1,y5,x2,y6,x2,y2,fill=darkenColor('#aa99ff',2),width=0)
        canvas.create_polygon(x4,y1,x4,y3,x3,y4,x3,y5,x2,y6,x2,y2,fill=darkenColor('#ff99aa',2),width=0)

    def redrawAll(self,canvas): # Manually implement animation using my own packed canvas
        self.canvas.delete(ALL)
        self.drawBackground(self.canvas)
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
        self.background = 0
        self.backgroundMax = self.app.sandboxMode.backgroundMax

    def modeActivated(self): # Scrap the graphics file canvas so that I can handle exceptions myself
        self.scrollMax = sorted(list(blockLibrary.values()),key=lambda x: x.y)[-1].y+2*Dragger.targetSize - 920
        self.app._canvas.pack_forget()
        self.canvas = Canvas(self.app._root,width=800,height=800)
        self.canvas.pack()
        TopLevel.globalCanvas = self.canvas
        TopLevel.curMode = 2
        for i in blockLibrary.values():
            i.y -= 140

    def modeDeactivated(self): # When deactivated, re-impose the graphics-file canvas
        self.canvas.pack_forget()
        self.app._canvas.pack()
        for i in blockLibrary.values():
            i.y += 140

    def pressingButtons(self,x,y): # Handling buttons in Function mode
        options = [self.app.splashMode,self.app.sandboxMode,self.app.splashMode]
        yVals = [625, 675, 725]
        for i in range(3):
            if (x > 590 or ((x-590)**2 + (y-yVals[i])**2) < 625) and x < 630 and y < yVals[i]+25 and y > yVals[i]-25:
                if i == 0: self.background += 1; self.background %= self.backgroundMax
                elif i == 1:
                    if popupBox('Type DONE to finish the function (no takebacks!)') == 'DONE':
                        if TopLevel.FuncSlots.slots[0] != None:
                            CreationMode.frankenstein.dataString = TopLevel.FuncSlots.slots[0].FuncToString()
                        else:
                            CreationMode.frankenstein.dataString = 'Null'
                        CreationMode.done = True
                        self.app.setActiveMode(self.app.sandboxMode)
                elif i == 2: self.app.setActiveMode(self.app.splashMode)

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
                if isinstance(contact,Dragger) and contact.exception():
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
        self.drawBackground(self.canvas)
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
        self.canvas.pack()
        TopLevel.globalCanvas = self.canvas
        self.mX, self.mY = 0,0

    def appStarted(self):
        self.hoverColor = 'White'
        self.spin = 0
        self.timerDelay = 5
        self.squares = [self.polarize(400,250,self.hexaSpin(200,math.radians(t)),math.radians(t)) for t in range(0,360,3)]

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

    def mouseMoved(self,event): # Used to detect hovering over buttons
        self.mX, self.mY = event.x, event.y

    def mousePressed(self,event):
        if event.x > 321 and event.x < 479:
            if event.y > 606 and event.y < 644: self.app.setActiveMode(self.app.sandboxMode)
            elif event.y > 656 and event.y < 694: self.app.setActiveMode(self.app.loadMode)
            elif event.y > 706 and event.y < 744: self.app.setActiveMode(self.app.tutorialMode)

    def timerFired(self): # Increment the animation
        self.spin += 5
        self.squares.append(self.squares.pop(0))
        red = int(85 * math.sin(2*math.pi*(5*self.spin/360)) + 170)
        green = int(85 * math.sin(2*math.pi*((5*self.spin+120)/360)) + 170)
        blue = int(85 * math.sin(2*math.pi*((5*self.spin+240)/360)) + 170)
        self.hoverColor = '#' + hex(red)[2:] + hex(green)[2:] + hex(blue)[2:]

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

    def buttonCords(self,x,y,w,h):
        x0, y0 = x - w, y + h
        x1, y1 = x + w, y + h
        x2, y2 = x + w, y - h
        x3, y3 = x - w, y - h
        return (x0-5,y0-5),(x0,y0),(x0+5,y0+5),(x1-5,y1+5),(x1,y1),(x1+5,y1-5),(x2+5,y2+5),(x2,y2),(x2-5,y2-5),(x3+5,y3-5),(x3,y3),(x3-5,y3+5)

    def drawButtons(self,canvas):
        if self.mX > 321 and self.mX < 479:
            if self.mY > 606 and self.mY < 644: colors = [self.hoverColor,'White','White',]
            elif self.mY > 656 and self.mY < 694: colors = ['White',self.hoverColor,'White']
            elif self.mY > 706 and self.mY < 744: colors = ['White','White',self.hoverColor]
            else: colors = ['White','White','White']
        else: colors = ['White','White','White']
        canvas.create_polygon(*self.buttonCords(400,625,75,15),smooth=True,fill=colors[0],width=None)
        canvas.create_text(400,625,text='Enter',fill='black',font='Times 26 bold')
        canvas.create_polygon(*self.buttonCords(400,675,75,15),smooth=True,fill=colors[1],width=None)
        canvas.create_text(400,675,text='Load',fill='black',font='Times 26 bold')
        canvas.create_polygon(*self.buttonCords(400,725,75,15),smooth=True,fill=colors[2],width=None)
        canvas.create_text(400,725,text='Tutorial',fill='black',font='Times 26 bold')

    def redrawAll(self,canvas): # Clear canvas with each tick & draw all squares
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(0,0,self.width,self.height,fill='Black')
        for i in range(6):
            sX, sY = self.squares[int(i*20-self.spin/2)%120]
            self.canvas.create_text((sX+800)/3,(sY+500)/3,font=f'Times {35-3*i}',fill='White',text='KIMCHI'[i])
        for i,square in enumerate(self.squares):
            self.drawSquare(9*i+self.spin%360,square,self.canvas)
        self.drawButtons(self.canvas)

class LoadMode(Mode):
    def modeActivated(self): # When activated, discard the graphics-file canvas
        self.app._canvas.pack_forget()
        self.canvas = Canvas(self.app._root,width=800,height=800)
        self.canvas.pack()
        TopLevel.globalCanvas = self.canvas
        self.allFiles = getKimchiFiles()
        self.carousel = 0
        self.kF = []
        self.popMax = 10
        loadSet = []
        for i in self.allFiles:
            loadSet.append(i)
            if len(loadSet) == self.popMax or i == self.allFiles[-1]:
                self.kF.append(loadSet)
                loadSet = []
        self.hovering = -1
        self.clicked = -1

    def appStarted(self):
        self.hoverColor = 'White'
        self.time = 0
        self.timerDelay = 5

    def modeDeactivated(self): # When deactivated, re-impose the graphics-file canvas
        self.canvas.pack_forget()
        self.app._canvas.pack()

    def mouseMoved(self,event):
        if event.x < 200 or event.x > 600 or event.y < 200 or event.y > 600 or len(self.kF) == 0:
            self.hovering = -1
            return
        for i in range(len(self.kF[self.carousel])):
            if event.y > 200+(i*400/self.popMax) and event.y < 200+((i+1)*400/self.popMax):
                self.hovering = i
                return
        self.hovering = -1

    def mousePressed(self,event):
        if event.x > 600 and event.x < 650 and event.y > 560 and event.y < 600:
            self.carousel += 1
            self.carousel %= len(self.kF)
        elif event.y > 600 and event.y < 650:
            if event.x > 200 and event.x < 350:
                self.app.setActiveMode(self.app.splashMode)
            elif event.x > 400 and event.x < 550 and self.clicked != -1:
                TopLevel.loadFile(self.kF[self.carousel][self.clicked])
                Sandbox.scrollMax = (len(blockLibrary)+len(customFunctions)) * Dragger.targetSize * 1.5
                self.app.setActiveMode(self.app.sandboxMode)
        elif event.x < 200 or event.x > 600 or event.y < 200 or event.y > 600:
            self.clicked = -1
            return
        else:
            if len(self.kF) == 0: return
            for i in range(len(self.kF[self.carousel])):
                if event.y > 200+(i*400/self.popMax) and event.y < 200+((i+1)*400/self.popMax):
                    self.clicked = i
                    return
        self.clicked = -1

    def timerFired(self): # Increment the animation
        self.time += 5
        red = int(55 * math.sin(2*math.pi*(5*self.time/360)) + 200)
        green = int(55 * math.sin(2*math.pi*((5*self.time+120)/360)) + 200)
        blue = int(55 * math.sin(2*math.pi*((5*self.time+240)/360)) + 200)
        self.hoverColor = '#' + hex(red)[2:] + hex(green)[2:] + hex(blue)[2:]

    def redrawAll(self,canvas):
        self.canvas.create_rectangle(0,0,self.width,self.height,fill='#888888')
        self.canvas.create_rectangle(200,200,self.width-200,self.height-200,fill='#bcbcbc')
        self.canvas.create_rectangle(200,150,500,200,fill='#343434')       
        self.canvas.create_text(210,175,font='Times 26 bold',text='Saved Kimchi Files:',anchor='w',fill='#efefef')
        if self.kF == []: self.canvas.create_text(400,250,font='Times 26 bold',text='There are no Kimchi Files to load')
        else:
            for i in range(len(self.kF[self.carousel])):
                if i == self.clicked: c = '#efefef'
                elif i == self.hovering: c = self.hoverColor
                else: c = None
                self.canvas.create_rectangle(200,200+(i*400/self.popMax),600,200+((i+1)*400/self.popMax),fill=c)
                self.canvas.create_text(210,200+((i+0.5)*400/self.popMax),text=self.kF[self.carousel][i],font='Times 26 bold',anchor='w')
        self.canvas.create_rectangle(200,600,350,650,fill='#efabab')
        self.canvas.create_text(275,625,font='Times 20 bold',text='Return Home')
        if self.clicked != -1:
            self.canvas.create_rectangle(375,600,525,650,fill='#abefab')
            self.canvas.create_text(450,625,font='Times 20 bold',text='Load File')
        if len(self.kF) > 1:
            self.canvas.create_rectangle(600,560,650,600,fill='#ababef')
            self.canvas.create_text(625,580,font='Times 20 bold',text='->')

class TutorialMode(Mode): # Help screen

    def appStarted(self):
        self.holding = [None]
        self.scrollY = self.scrollY2 = 0
        self.scrollM = 850
        self.prism = '#666666'
        self.time = 0
        self.textList = self.getText()
        self.my = 400
        self.scrollingUp = False
        self.scrollingDown = False
        self.timerDelay = 2 

    def modeActivated(self):
        TopLevel.curMode == 'T'
        letterSamples = 'abCdEFXyztTP'
        listSamples = ['Q',2,'S',None,'o',3,'V',0,'U',1,-1,10]
        controlSamples = [chr(955),1,2,3,4,5,6,chr(955),7,8,9,10]
        self.sampleBlocks = [ArithmeticFunction('Mystery',lambda x,y:(x**2+y**2)**0.5,100,700,False,[blockLibrary['0'] for i in range(2)],[None,None])]
        self.sampleBlocks += [LetterFunction('Quandry',lambda x: x.isupper(),100,910,False,[blockLibrary['Silent']],[None])]
        self.sampleBlocks += [IterableFunction('Enigma',lambda x,y,z: x.inject(y,z),100,1000,False,[blockLibrary['Empty'],blockLibrary['0'],blockLibrary['Null']],[None,None,None])]
        self.sampleBlocks += [PredicateFunction('Riddle',lambda x,y: (x and not y) or (y and not x),100,1150,False,[blockLibrary['False'] for i in range(2)],[None,None])]
        self.sampleBlocks += [ControlFunction('Secret',lambda x,y: kMap(x,y),100,1250,False,[blockLibrary['Null'],blockLibrary['Empty']],[None,None])]
        self.sampleBlocks += [Atom(i+1,200+60*(i%4),695+25*(i%3),False,f'{i+1}') for i in range(12)]
        self.sampleBlocks += [Atom(letterSamples[i],200+60*(i%4),895+25*(i%3),False,f'{letterSamples[i]}') for i in range(12)]
        self.sampleBlocks += [IterableFunction('3-list',lambda x,y,z: Axis.Axis(x,y,z),200,1000,False,[blockLibrary['Null'] for i in range(3)],[None,None,None])]
        self.sampleBlocks += [Atom(listSamples[i],300+60*(i%4),995+25*(i%3),False,f'{listSamples[i]}') for i in range(12)]
        self.sampleBlocks += [Atom(True,200,1155,False,'True'),Atom(True,300,1155,False,'True')]
        self.sampleBlocks += [Atom(False,200,1180,False,'False'),Atom(True,300,1180,False,'False')]
        self.sampleBlocks += [IterableFunction('3-list',lambda x,y,z: Axis.Axis(x,y,z),200,1250,False,[blockLibrary['Null'] for i in range(3)],[None,None,None])]
        self.sampleBlocks += [ArithmeticFunction('*',lambda x,y: x*y,200,1355,False,[blockLibrary['0'] for i in range(2)],[None,None])]
        self.sampleBlocks += [Atom(controlSamples[i],300+60*(i%4),1255+25*(i%3),False,f'{controlSamples[i]}') if controlSamples[i] != chr(955) else Atom(None,300+60*(i%4),1255+25*(i%3),False,f'{controlSamples[i]}') for i in range(12)]

    def timerFired(self): # Increment the animation
        self.time += 5
        red = int(55 * math.sin(2*math.pi*(5*self.time/360)) + 150)
        green = int(55 * math.sin(2*math.pi*((5*self.time+120)/360)) + 150)
        blue = int(55 * math.sin(2*math.pi*((5*self.time+240)/360)) + 150)
        self.prism = '#' + hex(red)[2:] + hex(green)[2:] + hex(blue)[2:]
        if self.scrollingUp and self.scrollY > 0: self.scrollY -= (100 - self.my)/10; self.scrollY2 -= (100 - self.my)/10
        if self.scrollingDown and self.scrollY < self.scrollM: self.scrollY += (self.my - 700)/10; self.scrollY2 += (self.my - 700)/10

    def getText(self): # Text for tutorial mode by coordinates
        out = []
        out.append({'x':200,'y':50,'text':'Welcome to','font':'Times 25 bold','anchor':'w','fill':'Black'})
        out.append({'x':335,'y':50,'text':'K.I.M.C.H.I.','font':'Times 30 bold','anchor':'w','fill':f'{self.prism}'})
        out.append({'x':310,'y':70,'text':'','font':'Times 30 bold','anchor':'w','fill':f'{self.prism}'})
        blocks1 = chunkifyString(f'''
        K.I.M.C.H.I. is a drag & drop programming language with an emphasis on evauluative programming. 
        K.I.M.C.H.I. operates by dragging 2 types of blocks: Atoms and Functions. Atoms are the fundemental 
        units that are used to construct other things (0,1,True,False,etc.)
        units that are used to construct other things (0,1,True,False,etc.)
        Functions are operations that have slots. The slots can be filled'
        with Atoms or other Functions, and when evaluated they will apply
        their operation to their slots. There are five types of Functions:''',90)
        blocks2 = '''
        Blue = arithmetic (mathematical operations like + and -)
        Green = letter (string opperations like ord and chr)
        Orange = iterable (list operations like len and range)
        Purple = predicatte (boolean operations like or and if-else)
        Yellow = control (higher order functions like map and filter)
        '''
        blocks3 = chunkifyString(f''' 
        There is a large pre-built library of functions to choose from, as
        well as the ability to use variables. There are {maxVars} pink V atoms
        representing each variable. The "Set Variables" Function in Sandbox Mode
        has slots that can be filled, and the value of whatever is inside of them
        is carried in the V atoms. Additionally, more functions can be made by
        pressing the green + button. You can make any arithmetic, letter,
        iterable or predicate Function you wish with between 1 and 9 inputs.
        While in Function Creator mode, the "Set Variables" slots will be'
        replaced by 1 slot representing the definition of the new function,
        When you are done, press the green + again to save the definition.
        To save your functions (and whatever you are doing in the Sandbox),
        click on the red "S" button to create a new save. When reloading
        K.I.M.C.H.I., press "LOAD" to find your saved file.
        ''',90)
        blocks4 = chunkifyString(f''' 
        Below is a function called Mystery that takes 2 inputs, along with
        Atoms representing the numbers from 1 to 12. Play around with them
        by move the Atoms (drag the red circle to pick it up) and slotting them
        into Mystery to see if you can figure out what it does.''',90)
        blocks5 = chunkifyString(f''' 
        Below are four more functions: Quandry, Enigma, Riddle and Secret. Each
        of them is accompanied by some Functions and Atoms that can be used
        to explore how they work. Note that each function has a default value
        if no value is provided.''',90)
        blocks6 = chunkifyString(f'''
        To clear out the blocks in the Variable Sandbox and wipe all Custom Functions,
        press the grey trash can button on the bottom right of the screen. To return
        to the home screen, press the blue house button (this will not erase the
        Custom Functions or the Variable Sandbox unless you exit out of K.I.M.C.H.I.
        without saving. To change the backdrop of the Variable Sandbox, press the
        yellow spiral button.''',90)
        blocks7 = chunkifyString(f'''
        PS: several functions involve randomness and random numbers. The random
        float/integer/item is reset upon clicking on the random function in question.
        Using random functions in higher order functions is ok in the Variibale Sandbox,
        but using random functions at all inside of custom functions is to be avoided.
        Instead, try to generate the randomness inside of inputs.''',90)
        for i in blocks1.splitlines() + blocks2.splitlines() + blocks3.splitlines() + [''] + blocks4.splitlines() + ([''] * 8) + blocks5.splitlines() + ([''] * 29) + blocks6.splitlines() + [''] + blocks7.splitlines():
            out.append({'x':30,'y':out[-1]['y']+20,'text':i,'font':'Times 16','anchor':'w','fill':'Black'})
        return out

    def mousePressed(self,event):
        if event.x > 750 and event.y < 60: self.app.setActiveMode(self.app.splashMode)
        for i in self.sampleBlocks:
            contact = i.touching(event.x,event.y,self)
            if contact != None:
                self.holding = contact
                contact.clickColor = '#2e9121' # Green circle if currently being held (no exception handling)
                if self.holding.parent != None:
                    for i in range(len(self.holding.parent.slots)):
                        if self.holding.parent.slots[i] == self.holding:
                            self.holding.parent.slots[i] = None
                    self.holding.parent.slotify()
                if contact not in self.sampleBlocks[:5]:
                    self.sampleBlocks.append(self.sampleBlocks.pop(self.sampleBlocks.index(contact)))
                return

    def mouseDragged(self,event):
        if self.holding == [None]: return
        self.holding.x = event.x
        self.holding.y = event.y + self.scrollY2
        if isinstance(self.holding,Function):
            self.holding.slotify()

    def mouseReleased(self,event): # Stop scrolling & release dragged Atoms/Funcitons
        if self.holding == [None]: return
        self.holding.x = event.x
        self.holding.y = event.y + self.scrollY2
        if isinstance(self.holding,Dragger):
            if isinstance(self.holding,Function): self.holding.slotify()
            for i in self.sampleBlocks:
                if isinstance(i,Function):
                    slot = i.insideSlot(event.x,event.y+self.scrollY2)
                    if slot != None:
                        i.slots[slot] = self.holding
                        self.holding.parent = i
                        i.slotify()
                        break
        self.holding = [None]

    def mouseMoved(self,event):
        self.my = event.y
        if event.y < 100 and self.scrollY > 0: self.scrollingUp = True; self.scrollingDown = False
        elif event.y > 700 and self.scrollY < self.scrollM: self.scrollingUp = False; self.scrollingDown = True
        else: self.scrollingUp = False; self.scrollingDown = False
    
    def drawDemoEval(self,canvas):
        operands = self.sampleBlocks[0].operands
        canvas.create_text(500,710-self.scrollY,anchor='w',text=f'Mystery({operands[0]},{operands[1]})',fill=self.prism,font='Times 20 bold')
        canvas.create_text(500,730-self.scrollY,anchor='w',text=f'= {self.sampleBlocks[0]()}',fill=self.prism,font='Times 20 bold')
        operands2 = self.sampleBlocks[1].operands
        canvas.create_text(500,910-self.scrollY,anchor='w',text=f'Quandry({operands2[0]})',fill=self.prism,font='Times 20 bold')
        canvas.create_text(500,930-self.scrollY,anchor='w',text=f'= {self.sampleBlocks[1]()}',fill=self.prism,font='Times 20 bold')
        operands3 = self.sampleBlocks[2].operands
        canvas.create_text(550,1010-self.scrollY,anchor='w',text=chunkifyString(f'Enigma({operands3[0]()},{operands3[1]},{operands3[2]})',27)+'\n'+chunkifyString(f'= {self.sampleBlocks[2]()}',27),fill=self.prism,font='Times 20 bold')
        operands4 = self.sampleBlocks[3].operands
        canvas.create_text(500,1160-self.scrollY,anchor='w',text=f'Riddle({operands4[0]},{operands4[1]})',fill=self.prism,font='Times 20 bold')
        canvas.create_text(500,1180-self.scrollY,anchor='w',text=f'= {self.sampleBlocks[3]()}',fill=self.prism,font='Times 20 bold')
        operands5 = self.sampleBlocks[4].operands
        canvas.create_text(550,1260-self.scrollY,anchor='w',text=chunkifyString(f'Riddle({operands5[0]},{operands5[1]()})',27)+'\n'+chunkifyString(f'= {self.sampleBlocks[4]()}',27),fill=self.prism,font='Times 20 bold')

    def redrawAll(self,canvas):
        for i in self.textList:
            tD = {'text': i['text'], 'font': i['font'], 'anchor': i['anchor'], 'fill': i['fill']}
            if i['text'] == 'K.I.M.C.H.I.': tD['fill'] = self.prism
            canvas.create_text(i['x'],i['y']-self.scrollY,**tD)
        for i in self.sampleBlocks:
            if isinstance(i,Function): i.drawFunction(canvas,self)
            else: i.drawAtom(canvas,self)
        x0, y2, x1, y3, xA, r = 750, 0, 800, 60, 775, 25
        xB, yB, xC, yC, xD, yD = 750, 740-self.scrollY, 770, 775-self.scrollY, 790, 700-self.scrollY
        canvas.create_polygon(xC-5,yC,xC+5,yC,xC+10,yD,xC-10,yD,fill=self.prism,width=0,smooth=True)
        canvas.create_polygon(xB,yB-10,xB,yB+10,xC,yC+10,xD,yB+10,xD,yB-10,xC,yC-10,fill=self.prism,width=0,smooth=True)
        canvas.create_rectangle(x0,y2,x1,y3,width=0,fill=darkenColor('#99aaff',2))
        canvas.create_oval(x0-r/1.5,y2,x0+r/1.5,y3,width=0,fill=darkenColor('#99aaff',2))
        canvas.create_polygon(x0,y2+20,x0,y3-10,x1-15,y3-10,x1-15,y2+20,xA-7,y2+10,width=0,fill='#99aaff')
        self.drawDemoEval(canvas) 

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
        TopLevel.VariableSlots = VarSetFunction('Set Variables',TopLevel.Vars)
        TopLevel.FuncSlots = GenesisFunction('Create Function')
        self.splashMode = SplashMode()
        self.creationMode = CreationMode()
        self.sandboxMode = Sandbox()
        self.loadMode = LoadMode()
        self.tutorialMode = TutorialMode()
        self.setActiveMode(self.splashMode)
        self.funcStrings = ''
        self.vBars, self.hBars = self.generateToothpickCurve(set([(400,400)]),set(),28)
        self.dragonCurve = self.generateDragonCurve([(370,250),(370,245)],11)
        self.polka = self.generatePolkaDots()
        self.diamonds = self.generateDiamonds()
        self.isoms = self.generateIsometries()
        self.kLogo = ImageTk.PhotoImage(self.scaleImage(self.loadImage('KLOGO.png'),0.25))

    def generateIsometries(self):
        out = []
        for x in range(100,740,120):
            for y in range(-10,820,20):
                if ((y+10)/2) % 20 == 0: out.append((x,y))
                else: out.append((x+60,y))
        return out

    def generatePolkaDots(self):
        return [(100+(35*i)%540,(37*i)%800,random.random()*25,self.polkaColor()) for i in range(60)]

    def polkaColor(self):
        if random.random() > 1/3:
            return rgb2hex(random.randint(0,100),random.randint(100,200),random.randint(150,250))
        return rgb2hex(random.randint(0,100),random.randint(150,250),random.randint(100,200))

    def generateDiamonds(self):
        return [(175+100*(i//20),20+(40*i)%800,random.random()*40-20,self.diamondColor()) for i in range(100)]

    def diamondColor(self):
        if random.random() > 1/3: return rgb2hex(random.randint(100,200),random.randint(50,150),random.randint(0,100))
        if random.random() > 1/3: return rgb2hex(random.randint(150,250),random.randint(50,150),random.randint(0,100))
        return rgb2hex(random.randint(150,250),random.randint(50,150),random.randint(0,100))

    def generateToothpickCurve(self,vBars,hBars,depth):
        if depth <= 0: return (vBars,hBars)
        newV, newH = vBars, hBars
        for v in vBars: 
            if (v[0],v[1]-16) not in newV.union(newH): newH.add((v[0],v[1]-8))
            if (v[0],v[1]+16) not in newV.union(newH): newH.add((v[0],v[1]+8))
        for h in hBars: 
            if (h[0]-16,h[1]) not in newV.union(newH): newV.add((h[0]-8,h[1]))
            if (h[0]+16,h[1]) not in newV.union(newH): newV.add((h[0]+8,h[1]))
        return self.generateToothpickCurve(newV,newH,depth-1)

    def generateDragonCurve(self,points,depth):
        if depth == 0: 
            return points
        out = points[:]
        for i in range(len(points)-2,-1,-1):
            out.append(self.rotatePoint(points[i],points[-1]))
        return self.generateDragonCurve(out,depth-1)

    def rotatePoint(self,point,center):
        x0, y0 = point
        xC, yC = center
        return (xC + (y0 - yC), yC - (x0 - xC))

    @staticmethod
    def newSave(): # Saves a new Kimchi File
        sName = popupBox('What do you want to save the file as?')
        if sName != None:
            if '.' in sName:
                nextStep = popupBox(f'"{sName}" is not a valid name.\nPress ok else to try again.')
                if nextStep != None: TopLevel.newSave()
            elif sName in getKimchiFiles():
                nextStep = popupBox(f'"{sName}" is already a file name.\nSaving will overwrite it.\nType "ok" to proceed.')
                if nextStep == 'ok':
                    os.remove(f"<<KimchiFile>> {sName}.txt")
                    f = open(f"<<KimchiFile>> {sName}.txt","w+")
                    for i in customFunctions:
                        f.write('f: '+str(i.exportFunction())+'\n')
                    f.write(Dragger.exportClones())
                    f.close()
            else:
                f = open(f"<<KimchiFile>> {sName}.txt","w+")
                for i in customFunctions:
                    f.write('f: '+str(i.exportFunction())+'\n')
                f.write(Dragger.exportClones())
                f.close()

    @staticmethod
    def loadFile(name): # Loads a Kimchi File
        try:
            TopLevel.wipeBlocks()
            f = open(f"<<KimchiFile>> {name}.txt","r")
            r = f.read()
            for i in r.splitlines():
                if i[0:3] == 'f: ':
                    t = eval(i.strip('f: '))
                    if t[0] not in blockLibrary:
                        CustomFunction.uberGenesis(t)
                if i[0:3] == 'v: ':
                    t = eval(i.strip('v: '))
                    if t[0] == 'Set Variables':
                        TopLevel.VariableSlots = VarSetFunction('Set Variables',list(map(lambda x: Dragger.superGenesis(x),t[3:])))
                        TopLevel.Vars = TopLevel.VariableSlots.slots
                    else: Dragger.superGenesis(t)
            TopLevel.VariableSlots.slotify()
            TopLevel.VariableSlots.getDimensions()
            f.close()
        except Exception as e:
            print(e)
            pass

    @staticmethod
    def globalException(e): # What to put on screen instead of an exception that crashes TKinter
        x0, y0 = TopLevel.width * 0.825, TopLevel.height * 0.8625
        TopLevel.globalCanvas.create_oval(x0,y0,x0+20,y0+20,fill='#f2e422',width=0)
        TopLevel.globalCanvas.create_text(x0,y0+25,text=chunkifyString(customError(e),25),anchor='nw',font='Times 12',fill='White')

    @staticmethod
    def generateAtoms(atomX): # Generates all Atoms
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

    @staticmethod
    def generateFunctions(atomX): # Generates all pre-built functions from a .txt file
        funcStr = open('PreFunctions.txt','r')
        iFunctions = funcStr.read().splitlines()
        funcStr.close()
        out = [Function.genesis(iFunctions[0],atomX,Dragger.targetSize*2*(len(blockLibrary)+0.75))]
        for i in iFunctions[1:]:
            out += [Function.genesis(i,atomX,out[-1].y+Dragger.targetSize*2)]

    @staticmethod
    def generateIO(atomX): # Generates function input Atoms
        out = [IOAtom(None,atomX,15,True,'I0')]
        out += [IOAtom(None,atomX,out[-1].y+(Dragger.targetSize*2*i),True,'I'+str(i)) for i in range(1,9)]

    @staticmethod
    def wipeBlocks(): # Clears out all blocks
        global blockLibrary 
        global ioLibrary 
        global customFunctions
        blockLibrary = {}
        ioLirary = {}
        customFunctions = set()
        Dragger.cloneList = []
        Dragger.cloneList2 = []
        TopLevel.VariableSlots.wipeVariables()
        TopLevel.generateAtoms(25)
        TopLevel.generateIO(25)
        TopLevel.generateFunctions(25)

TopLevel(width=800,height=800)
