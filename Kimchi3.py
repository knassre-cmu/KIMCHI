import copy, math, random, string, os
from tkinter import *
import Axis
from cmu_112_graphics import *

blockLibrary = {} # Dictionary of all known functions/atoms (used for defaults & drawing function library)
maxVars = 32

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
        if value == cases[i]():
            return outcomes[i]
    return blockLibrary['Null']

def minimize(f,l): # Loop over a list and pick out the item with the smallest output to some function
    maxInput = blockLibrary['Null']
    maxOutput = blockLibrary['∞']
    for i in l:
        hof = replaceLambda(f,i)()
        if fhof < maxOutput:
            maxInput = i
            maxOutput = hof
    return maxInput

def optimize(f,l): # Loop over a list and pick out the item with the largest output to some function
    maxInput = blockLibrary['Null']
    maxOutput = blockLibrary['-∞']
    for i in l:
        hof = replaceLambda(f,i)()
        if fhof > maxOutput:
            maxInput = i
            maxOutput = hof
    return maxInput

def findFirst(f,l): # Loop over a list and pick out the first item that returns true for some function
    for i in l:
        if replaceLambda(f,i)():
            return i
    return blockLibrary['Null']

def get(t,d,default): # Treat a list of 2-item lists as a dicitonary with the .get() method
    for i in d:
        if i()[0] == t:
            return i()[1]
    return default

def kSort(f,l): # Sorting HOF
    return

def kMap(f,l): # Mapping HOF
    out = []
    for i in l:
        out.append(replaceLambda(f,i))
    return Axis.Axis(*out)

def kFilter(f,l):
    out = []
    for i in l:
        if replaceLambda(f,i):
            out.append(i)
    return Axis.Axis(*out)

def replaceLambda(f,i):
    if f.name == chr(955):
        return i
    elif isinstance(f,Function):
        return f.value(*map(lambda x: replaceLambda(x,i),f.operands))
    return f()

class Dragger(object): # Class of all draggable blocks

    cloneList = [] # List of all block clones
    fontSize = 18
    targetSize = 10
    
    def __bool__(self): # If treated as a boolean, return boolean of value
        return bool(self())

    def touching(self,x,y,app): # Is this block being touched
        if self.cloneable: # If cloneable, create a clone (and adjust for scrolling)
            if (x-self.x)**2 + (y+app.scrollY-self.y)**2 < 100:
                if type(self) == Atom:
                    return Atom(self.value,x,y,False,self.name)
                return eval(self.fType)(self.name,self.value,x,y,False,self.defaults,[None for i in range(len(self.operands))])
        elif (x-self.x)**2 + (y-self.y)**2 < 100: # Else, just move it (and bring to top)
            Dragger.cloneList.remove(self)
            Dragger.cloneList.append(self)
            return self

    def __eq__(self,other):
        return type(self) == type(other) and self() == other() and self.x == other.x and self.y == other.y

class Function(Dragger): # Class of all functions
    
    def __init__(self,fType,name,value,x,y,cloneable,defaults,slots,parent=None):
        self.fType, self.name, self.value, self.x, self.y = fType, name, value, x, y
        self.cloneable, self.defaults, self.slots = cloneable, defaults, slots
        self.parent = parent
        self.getDimensions()
        self.operands = slots[:]
        self.getOperands()
        if self.cloneable: blockLibrary[self.name] = self
        else: Dragger.cloneList.append(self)

    def getDimensions(self): # Update dimensions as things are dragged in/out
        self.xMin = Dragger.targetSize + 10 + len(self.name) * Dragger.fontSize/2.3
        self.xMax = max([self.xMin]+list(map(lambda x: 0 if x == None else x.xMin, self.slots)))
        self.yMin = 2 * Dragger.targetSize
        self.yCords = [0]
        for i in range(len(self.slots)):
            if self.slots[i] == None or not isinstance(self.slots[i-1],Dragger): self.yCords.append(self.yCords[-1]+self.yMin*1.1)
            else: self.yCords.append(self.yCords[-1]+self.slots[i-1].ySize)
        self.ySize = self.yCords[-1] + self.yMin*1.1

    def getOperands(self): # Update operands according to if slots are empty or not
        for i in range(len(self.slots)):
            if self.slots[i] == None: self.operands[i] = self.defaults[i]
            else: self.operands[i] = self.slots[i]

    def __call__(self): # If being treated like a function, call the lambda expression
        try:
            return self.value(*map(lambda x: x(), self.operands))
        except Exception as e: # If inputs are invalid, return the defualt and print an error
            TopLevel.globalException(e)
            return self.value(*map(lambda x: x(), self.defaults))

    def __repr__(self): # How to print for debugging:
        return f'{self.name} ({", ".join(map(lambda x: str(x),self.operands))}) => {str(self())}'

    def drawLibraryFunction(self,canvas,app): # Draws the cloneable functions in the library
        clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + Dragger.targetSize + self.xMax, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x,y0-app.scrollY,x1,y1-app.scrollY,fill=self.color,width=0)
        canvas.create_oval(x0,y0-app.scrollY,x2,y1-app.scrollY,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY,anchor='w',text=self.name,font=f'Arial {Dragger.fontSize}')

    def drawFunction(self,canvas,app): # Draws the cloned functions
        if self == app.holding: clickColor = '#2e9121' # Green circle if currently being held
        else: clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + self.xMin + Dragger.targetSize, self.y + Dragger.targetSize
        y2 = y1 + self.ySize - self.yMin
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x,y0,x1,y2,fill=self.color,width=0)
        canvas.create_oval(x0,y0,x2,y1,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y,anchor='w',text=self.name,font=f'Arial {Dragger.fontSize}')
        for i in range(len(self.operands)):
            if self.slots[i] == None:
                yCord = self.yCords[i]
                canvas.create_oval(self.x+5,y1+yCord,self.x+5+Dragger.targetSize*2,y1+yCord+Dragger.targetSize*2,fill='White',width=0)
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
        if abs(x-self.x-15) >= 10:
            return
        for r in range(1,len(self.yCords)):
            i = self.yCords[r]
            if abs(i+self.y-y) < 10:
                return r-1

    def slotify(self,allowUp=True): # Adjust x & y coordiantes of all children in refrence to parent
        for i in range(1,len(self.yCords)):
            if self.slots[i-1] != None:
                self.slots[i-1].y = self.y + self.yCords[i] - 3
                self.slots[i-1].x = self.x + 15
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
                Dragger.cloneList.remove(child)

class ArithmeticFunction(Function):
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('ArithmeticFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#49bcdf'

class LetterFunction(Function):
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('LetterFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#94df49'

class PredicateFunction(Function):
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('PredicateFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#9079f6'

class IterableFunction(Function):
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('IterableFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#ea5639'

class ControlFunction(Function):
    def __init__(self,name,value,x,y,cloneable,defaults,slots,parent=None):
        super().__init__('ControlFunction',name,value,x,y,cloneable,defaults,slots,parent)
        self.color = '#e1ae5b'

    def __call__(self): # If being treated like a function, call the lambda expression
        try:
            return self.value(self.operands[0],self.operands[1]())
        except Exception as e: # If inputs are invalid, return the defualt and print an error
            TopLevel.globalException(e)
            return self.value(*map(lambda x: x(), self.defaults))

class Atom(Dragger): # Class of all atoms
    def __init__(self,value,x,y,cloneable,name,parent=None):
        self.value, self.x, self.y, self.cloneable, self.name, self.parent = value, x, y, cloneable, name, parent
        self.xMin = self.xMax = Dragger.targetSize + 10 + len(self.name) * Dragger.fontSize/2.3
        self.yMin = self.ySize = 2 * Dragger.targetSize
        self.getColor()
        if cloneable: blockLibrary[self.name] = self
        else: Dragger.cloneList.append(self)

    def getColor(self): # Does exactly what is sounds like
        if isinstance(self.value,(int,float)):
            if isinstance(self.value,bool): self.color = '#9079f6'
            else: self.color = '#49bcdf'
        elif isinstance(self.value,str): self.color = '#94df49'
        elif isinstance(self.value,Axis.Axis): self.color = '#ea5639'
        else: self.color = '#b7abbf'

    def __call__(self): # If treated as a function, return the value
        return self.value

    def __repr__(self): # How to print for debugging
        return str(self())

    def drawLibraryAtom(self,canvas,app): # Draws the cloneable atoms in the library
        clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + self.xMin + Dragger.targetSize, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x,y0-app.scrollY,x1,y1-app.scrollY,fill=self.color,width=0)
        canvas.create_oval(x0,y0-app.scrollY,x2,y1-app.scrollY,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y-app.scrollY,anchor='w',text=self.name,font=f'Arial {Dragger.fontSize}')

    def drawAtom(self,canvas,app): # Draws the cloned atoms
        if self == app.holding: clickColor = '#2e9121' # Green circle if currently being held
        else: clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x - Dragger.targetSize, self.y - Dragger.targetSize
        x1, y1 = self.x + Dragger.targetSize + self.xMin, self.y + Dragger.targetSize
        x2 = self.x + Dragger.targetSize
        canvas.create_rectangle(self.x,y0,x1,y1,fill=self.color,width=0)
        canvas.create_oval(x0,y0,x2,y1,fill=clickColor,width=0)
        canvas.create_text(x2+3,self.y,anchor='w',text=self.name,font=f'Arial {Dragger.fontSize}')

class Sandbox(Mode): # Mode with the sandbox for playing with Atoms & Functions
    def appStarted(self):
        self.holding = [None]
        self.scrollY = 5
        self.scrollMax = len(blockLibrary) * Dragger.targetSize * 1.43
        self.scrollHeight = self.height
        self.scrolling = False

    def modeActivated(self): # Scrap the graphics file canvas so that I can handle exceptions myself
        self.app._canvas.pack_forget()
        self.canvas = Canvas(self.app._root,width=800,height=800)
        self.canvas.config(width=800,height=800)
        self.canvas.pack()
        TopLevel.globalCanvas = self.canvas

    def modeDeactivated(self): # When deactivated, re-impose the graphics-file canvas
        self.canvas.pack_forget()
        self.app._canvas.pack()

    def mousePressed(self,event): # Check for scrolling & dragging Atoms/Functions
        if event.x < 10:
            self.scrolling = True
            return
        for i in list(blockLibrary.values()) + Dragger.cloneList:
            if i == None:
                continue
            contact = i.touching(event.x,event.y,self)
            if contact != None:
                self.holding = contact
                if self.holding.parent != None:
                    for i in range(len(self.holding.parent.slots)):
                        if self.holding.parent.slots[i] == self.holding:
                            self.holding.parent.slots[i] = None
                    self.holding.parent.slotify()
                print(self.holding)
                return

    def mouseDragged(self,event): # Move an Atom/Funciton if it is being dragged, or adjust scrollbar
        if self.scrolling:
            self.scrollY = event.y/self.width * self.scrollMax
        if self.holding == [None]:
            return
        self.holding.x = event.x
        self.holding.y = event.y
        if isinstance(self.holding,Function):
            self.holding.slotify()

    def mouseReleased(self,event): # Stop scrolling & release dragged Atoms/Funcitons
        self.scrolling = False
        if self.holding == [None]:
            return
        self.holding.x = event.x
        self.holding.y = event.y
        xEdge = event.x + self.holding.xMax
        if event.x - Dragger.targetSize < 40 or xEdge > self.width*0.8 or abs(event.y-self.height/2) > -2 + self.height/2:
            Dragger.cloneList.remove(self.holding)
            if isinstance(self.holding,Function):
                self.holding.killChildren()
        else:
            for i in Dragger.cloneList:
                if isinstance(i,Function):
                    slot = i.insideSlot(event.x,event.y)
                    if slot != None:
                        i.slots[slot] = self.holding
                        self.holding.parent = i
                        i.slotify()
                        break
        self.holding = [None]

    def drawLibraryScroll(self,canvas): # Draw the scrollbar
        x0, y0 = 0, 0
        x1, y1 = 15, self.height
        y2 = y1 * self.scrollY / self.scrollMax
        canvas.create_rectangle(x0,y0,x1,y1,fill='#d1d1d2',width=0)
        canvas.create_oval(4,y2-7,14,y2+7,fill='#a8a7a9',width=0)

    def redrawAll(self,canvas): # Manually implement animation using my own packed canvas
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(0,0,self.width*0.8,self.height,fill='Black',width=0)
        self.drawLibraryScroll(self.canvas)
        for book in blockLibrary.values():
            if isinstance(book,Atom):
                book.drawLibraryAtom(self.canvas,self)
            else:
                book.drawLibraryFunction(self.canvas,self)
        for clone in Dragger.cloneList:
            if isinstance(clone,Atom):
                clone.drawAtom(self.canvas,self)
            else:
                clone.drawFunction(self.canvas,self)

class TopLevel(ModalApp): # Outermost app class
    globalCanvas = None

    def appStarted(self):
        self.generateAtoms(25)
        self.generateFunctions(25)
        #self.splashMode = SplashMode()
        self.sandboxMode = Sandbox()
        self.setActiveMode(self.sandboxMode)
    
    @staticmethod
    def globalException(e): # What to put on screen instead of an exception that crashes TKinter
        TopLevel.globalCanvas.create_text(200,50+len(str(e)),text=str(e),fill='Grey',anchor='w')

    def generateAtoms(self,atomX): # Generates all Atoms
        out = [Atom(0,atomX,15,True,'0')]
        out += [Atom(i,atomX,out[-1].y+(Dragger.targetSize*2*i),True,str(i)) for i in range(1,11)]
        out += [Atom(-1,atomX,out[-1].y+Dragger.targetSize*2,True,'-1')]
        out += [Atom(math.tau,atomX,out[-1].y+Dragger.targetSize*2,True,chr(428))]
        out += [Atom(math.e,atomX,out[-1].y+Dragger.targetSize*2,True,'e')]
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

    def generateFunctions(self,atomX): # Generates all pre-built functions from a .txt file
        funcStr = open('PreFunctions.txt','r')
        iFunctions = funcStr.read().splitlines()
        funcStr.close()
        out = [Function.genesis(iFunctions[0],atomX,Dragger.targetSize*2*(len(blockLibrary)+0.75))]
        for i in iFunctions[1:]:
            out += [Function.genesis(i,atomX,out[-1].y+Dragger.targetSize*2)]

TopLevel(width=800,height=800)
