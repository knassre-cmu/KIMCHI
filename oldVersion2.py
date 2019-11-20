# Most recent file before switching over to the version 3 (bears lots of similarity to the code submitted in TP1)
# Includes graphics, modes, and functions in a library that were interpreted from strings.
# Class structure not well suited to slotting in/out, and very slow on the graphics front.

import copy, math, random, string, os
from tkinter import *
import Axis
from cmu_112_graphics import *

# Place ALL top-level calls under a try-except which will display errors on canvas
def globalException(exceptype,name,params=[]):
    if exceptype == 'n': # Argument number error
        print(f'"{name}" recieved {params[0]} arguments (expected {params[1]})')
    if exceptype == 't': # Argument type error
        print(f'"{name}" cannot operate on {params[0]} (expected {params[1]})')

globalLibrary = {} # Dictionary of all known Functions / Atom

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
    return globalLibrary['Null']

def minimize(f,l): # Loop over a list and pick out the item with the smallest output to some function
    maxInput = globalLibrary['Null']
    maxOutput = globalLibrary['∞']
    for i in l:
        if f(i) < maxOutput:
            maxInput = i
            maxOutput = f(i)
    return maxInput

def optimize(f,l): # Loop over a list and pick out the item with the largest output to some function
    maxInput = globalLibrary['Null']
    maxOutput = globalLibrary['-∞']
    for i in l:
        if f(i) > maxOutput:
            maxInput = i
            maxOutput = f(i)
    return maxInput

def findFirst(f,l): # Loop over a list and pick out the first item that returns true for some function
    for i in l:
        if f(i):
            return i
    return globalLibrary['Null']

def get(t,d,default): # Treat a list of 2-item lists as a dicitonary with the .get() method
    for i in d:
        if i()[0] == t:
            return i()[1]
    return default

# Class for all draggable things
class Dragger():

    blockRecord = [] # List of all non-clones

    def __bool__(self): # How to treat as a boolean, depending on if it is a Function or Atom
        if isinstance(self,Function):
            return bool(self())
        return bool(self.value)

    def __repr__(self): # How to treat as a string, depending on if it is a Function or Atom
        if isinstance(self,Atom):
            return str(self.value)
        elif isinstance(self,Function):
            return (f'Function: {self.name} ({", ".join(map(lambda x: str(x),self.operands))}) => {self()}')
    
    def touching(self,x,y,user): # Check if the mouse is touching the red circle of a Function/Atom
        if self.cloneable: # If cloneable, create a clone (and adjust for scrolling)
            if (x-self.x)**2 + (y+user.scrollY-self.y)**2 < 64:
                if isinstance(self,Function):
                    user.holding = Function(self.typ,self.name,self.value,x,y,False,self.operands)
                    return True
                if isinstance(self,Atom):
                    user.holding = Atom(self.value,x,y,False,self.name)
                    return True
        elif (x-self.x)**2 + (y-self.y)**2 < 64: # If an ordinary block, just move it
            self.popToTop()
            user.holding = self
            return True

    def popToTop(self):
        i = 0
        x, y = self.x, self.y
        for i,b in filter(lambda x: x[1] == self, enumerate(Dragger.blockRecord)):
            if (x, y) == (b.x, b.y):
                Dragger.blockRecord.append(Dragger.blockRecord.pop(i))
                return

    def deleteBlock(self):
        i = 0
        x, y = self.x, self.y
        for i,b in filter(lambda x: x[1] == self, enumerate(Dragger.blockRecord)):
            if (x, y) == (b.x, b.y):
                Dragger.blockRecord.pop(i)
                return

class Function(Dragger): # Class for all functions

    FunctionID = 0 # Counter thatt increases with each created function -> used to ID them

    def __init__(self,typ,name,value,x,y,cloneable=False,operands=[]):
        self.x, self.y, self.cloneable, self.value = x, y, cloneable, value
        self.typ, self.name, self.size = typ, name, 5 + len(name) * 5
        self.ySize = 9
        self.accordian = self.ySize * len(operands)
        print(typ,name)
        if typ[0] == 'a':
            self.color = '#49bcdf'
        elif typ[0] == 'l':
            self.color = '#94df49'
        elif typ[0] == 'i':
            self.color = '#ea5639'
        elif typ[0] == 'p':
            self.color = '#9079f6'
        elif typ[0] == 'c':
            self.color = '#e1ae5b'
        self.args = len(value.__code__.co_varnames)
        self.operands = operands + [[] for i in range(self.args-len(operands))]
        self.value = value
        self.ID = Function.FunctionID # Helps prevent an unusual bug where dragging a cloned
        Function.FunctionID += 1      # block will also drag an identical clone nearby
        if cloneable:
            globalLibrary[self.name] = self
        else:
            Dragger.blockRecord.append(self)

    def __eq__(self,other): # How to compare Functions to other things
        if self.cloneable:
            return False
        if isinstance(other,Atom):
            return self() == other.value
        elif isinstance(other,Function):
            return self() == other()
        return self() == other

    def __call__(self): # Calling a Function like an actual function
        if type(self.value) == Atom:
            return self.value.value
        elif len(self.operands) != self.args:
            globalException('n',self.name,[len(self.operands),self.args])
            return None
        else:
            return Atom(self.value(*map(lambda x: x(), self.operands)))

    def genesis(s,x,y): # Generates a new functiton from a string
        parts = tuple(s.split(' | '))
        typ, name, func, defaultNames = parts
        func = eval(func)
        defaults = list(map(lambda x: globalLibrary[x], defaultNames.split(' ')))
        return Function(typ.lower()[:-8],name,func,x,y,True,defaults)

    def drawFunction(self,canvas,user): # Drawing each function
        if self == user.holding and self.x == user.holding.x and self.y == user.holding.y:
            clickColor = '#2e9121' # Green circle if currently being held
        else:
            clickColor = '#b91613' # Red circle if not
        x0, y0 = self.x-4.5, self.y - self.ySize/2
        x1, y1 = self.x + self.size + 9.5, self.y + self.ySize/2
        if self.cloneable: # Adjust for scrollbar if a cloneable
            canvas.create_rectangle(self.x,y0-user.scrollY,x1,y1-user.scrollY,fill=self.color,width=0)
            canvas.create_oval(x0,y0-user.scrollY,x0+9,y0+9-user.scrollY,fill=clickColor,width=0)
            canvas.create_text(x1-self.size,self.y-user.scrollY,anchor='w',text=self.name,font='Arial 9')
        else:
            y2 = y1 + self.accordian
            canvas.create_rectangle(self.x,y0,x1,y2,fill=self.color,width=0)
            canvas.create_oval(x0,y0,x0+9,y0+9,fill=clickColor,width=0)
            canvas.create_text(x1-self.size,self.y,anchor='w',text=self.name,font='Arial 9')
            for i in range(len(self.operands)):
                canvas.create_rectangle(x0+6,y1+9*i,x1-2,y1+9*(i+0.8),fill='White',width=0)

class Atom(Dragger): # Class for all Atoms (leaf elements that can be used to construct 
    
    AtomID = 0
    atomLibrary = []

    def __init__(self,value,x=0,y=0,cloneable=False,name=None):
        self.x, self.y, self.cloneable = x, y, cloneable
        self.value, self.name = value, name
        self.ySize = 9
        if self.name in ['space', 'Silent', 'Null', 'Empty', '∞', '-∞'] or self.value in [math.e, math.tau]:
            self.textVersion = self.name
        else:
            self.textVersion = str(self.value)
        self.size = 5 + len(self.textVersion) * 5
        if isinstance(value,(int,float)):
            if isinstance(value,bool):
                self.color = '#9079f6'
            else:
                self.color = '#49bcdf'
        elif isinstance(value,str):
            self.color = '#94df49'
        elif isinstance(value,Axis.Axis):
            self.color = '#ea5639'
        else:
            self.color = '#b7abbf'
        self.ID = Atom.AtomID
        Atom.AtomID += 1
        if cloneable:
            globalLibrary[self.textVersion] = self
        else:
            Dragger.blockRecord.append(self)
         
    def __call__(self): # If an Atom is called, return associated value
        return self.value
            
    def __eq__(self,other): # How to compare Atoms to other things
        if self.cloneable:
            return False
        if isinstance(other,Atom):
            return self.value == other.value
        elif isinstance(other,Function):
            return self.value == other()
        else:
            return self.value == other

    def drawAtom(self,canvas,user): # Drawing each Atom
        if self == user.holding and self.x == user.holding.x and self.y == user.holding.y:
            clickColor = '#2e9121' # Same as before
        else:
            clickColor = '#b91613'
        x0, y0 = self.x-4.5, self.y - self.ySize / 2
        x1, y1 = self.x + self.size + 9.5, self.y + self.ySize / 2
        if self.cloneable: # Same as before
            canvas.create_rectangle(self.x,y0-user.scrollY,x1,y1-user.scrollY,fill=self.color,width=0)
            canvas.create_oval(x0,y0-user.scrollY,x0+9,y0+9-user.scrollY,fill=clickColor,width=0)
            canvas.create_text(x1-self.size,self.y-user.scrollY,anchor='w',text=self.textVersion,font='Arial 9')
        else:
            canvas.create_rectangle(self.x,y0,x1,y1,fill=self.color,width=0)
            canvas.create_oval(x0,y0,x0+9,y0+9,fill=clickColor,width=0)
            canvas.create_text(x1-self.size,self.y,anchor='w',text=self.textVersion,font='Arial 9')

class SplashMode(Mode): # The colorful splash screen mode
    def modeActivated(self): # When activated, discard the graphics-file canvas
        self.app._canvas.pack_forget()
        self.canvas = Canvas(self.app._root,width=800,height=800)
        self.canvas.config(width=800,height=800)
        self.canvas.pack()

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
        if event.key == 'e':
            self.app.setActiveMode(self.app.mainMode)

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
        for i,square in enumerate(self.squares):
            self.drawSquare(9*i+self.spin%360,square,self.canvas)

class MainMode(Mode): # Mode with the sandbox for playing with Atoms & Functions
    def modeActivated(self): # Same as above
        self.app._canvas.pack_forget()
        self.canvas = Canvas(self.app._root,width=800,height=800)
        self.canvas.config(width=800,height=800)
        self.canvas.pack()

    def appStarted(self):
        self.holding = [None]
        self.scrollY = 5
        self.scrollMax = 900
        self.scrollHeight = 480
        self.scrolling = False

    def modeDeactivated(self): # Same as above
        self.canvas.pack_forget()
        self.app._canvas.pack()

    def keyPressed(self,event): # Switch betwene modes (temporary)
        if event.key == 'e':
            self.app.setActiveMode(self.app.splashMode)

    def mousePressed(self,event): # Check for scrolling & dragging Atoms/Functions
        if event.x < 10:
            self.scrolling = True
        for i in list(globalLibrary.values()) + Dragger.blockRecord:
            if i == None or (i.cloneable and i.y - self.scrollY > self.scrollHeight):
                continue
            if i.touching(event.x,event.y,self):
                print(i)
                break

    def mouseDragged(self,event): # Move an Atom/Funciton if it is being dragged, or adjust scrollbar
        if self.scrolling:
            self.scrollY = event.y/self.width * self.scrollMax
        if self.holding == [None]:
            return
        self.holding.x = event.x
        self.holding.y = event.y

    def mouseReleased(self,event): # Stop scrolling & release dragged Atoms/Funcitons
        self.scrolling = False
        if self.holding == [None]:
            return
        self.holding.x = event.x
        self.holding.y = event.y
        xEdge = event.x + self.holding.size + 9.5
        if event.x - 4.5 < 40 or xEdge > 600 or abs(event.y-400) > 395.5:
            self.holding.deleteBlock()
        self.holding = [None]

    def drawLibraryScroll(self,canvas): # Draw the scrollbar
        x0, y0 = 0, 0
        x1, y1 = 15, self.height
        y2 = y1 * self.scrollY / self.scrollMax
        canvas.create_rectangle(x0,y0,x1,y1,fill='#d1d1d2',width=0)
        canvas.create_oval(4,y2-7,14,y2+7,fill='#a8a7a9',width=0)

    def drawAddFunction(self,canvas): # Draw the add-function button
        x0, y0, x1, y1 = 560, 707, 600, 747
        x2, y2 = (x0 + x1) / 2, (y0 + y1) / 2
        canvas.create_rectangle(x2-5,y0,x1,y1,fill='#19ac0c',width=0)
        canvas.create_oval(x0-5,y0,x1,y1,fill='#19ac0c',width=0)
        canvas.create_rectangle(x2-3,y0+3,x2+3,y1-3,fill='#5cf34f',width=0)
        canvas.create_rectangle(x0+3,y2-3,x1-3,y2+3,fill='#5cf34f',width=0)

    def drawReturnHome(self,canvas): # Draw the return-home button
        x0, y0, x1, y1 = 560, 747, 600, 787
        x2, y2 = (x0 + x1) / 2, (y0 + y1) / 2
        canvas.create_rectangle(x2-5,y0,x1,y1,fill='#5760e0',width=0)
        canvas.create_oval(x0-5,y0,x1,y1,fill='#5760e0',width=0)
        canvas.create_polygon(x0+10,y2-3,x2,y2-15,x1-10,y2-3,fill='#a5a9ee',width=0)
        canvas.create_rectangle(x0+10,y2-3,x1-10,y1-7,fill='#a5a9ee',width=0)
        canvas.create_rectangle(x0+15,y2+2,x1-15,y1-12,fill='#5760e0',width=0)

    def drawAnimationMode(self,canvas): # Draw the add-function button
        x0, y0, x1, y1 = 560, 667, 600, 707
        x2, y2 = (x0 + x1) / 2, (y0 + y1) / 2
        xA, yA = x2 - 2, y2 - 18
        xB, yB = x2 + 8, y2 - 12
        xC, yC = x2 - 14, y2 - 6
        xD, yD = x2 + 20, y2 - 0
        xE, yE = x2 - 20, y2 + 6
        xF, yF = x2 + 10, y2 + 12
        xG, yG = x2 + 4, y2 + 16
        canvas.create_rectangle(x2-5,y0,x1,y1,fill='#db3d3d',width=0)
        canvas.create_oval(x0-5,y0,x1,y1,fill='#db3d3d',width=0)
        canvas.create_line(xA,yA,xB,yB,xC,yC,xD,yD,xE,yE,xF,yF,xG,yG,fill='#e98686',width=5,smooth=True)

    def redrawAll(self,canvas): # Draw buttons, Atom/Function library, and all Atoms/Functions in the sandbox
        self.canvas.delete(ALL)
        self.canvas.create_rectangle(0,0,600,self.height,fill='Black',width=0)
        self.drawReturnHome(self.canvas)
        self.drawAnimationMode(self.canvas)
        self.drawAddFunction(self.canvas)
        self.drawLibraryScroll(self.canvas)
        for b in list(globalLibrary.values()) + Dragger.blockRecord:
            if b == None:
                continue
            if b.y - self.scrollY > 9 and b.y - self.scrollY <= self.scrollHeight or not b.cloneable:
                if isinstance(b,Atom):
                    b.drawAtom(self.canvas,self)
                elif isinstance(b,Function):
                    b.drawFunction(self.canvas,self)

class TopLevel(ModalApp): # Outermost app class
    def appStarted(self):
        self.splashMode = SplashMode()
        self.mainMode = MainMode()
        self.setActiveMode(self.splashMode)
        atomX = 20
        self.generateAtoms(atomX)
        self.generateFunctions(atomX)

    def generateAtoms(self,atomX): # Generates all Atoms
        ySize = 10
        out = [Atom(0,atomX,15,True)]
        out += [Atom(i,atomX,out[-1].y+(ySize*i),True) for i in range(1,11)]
        out += [Atom(-1,atomX,out[-1].y+ySize,True)]
        out += [Atom(math.tau,atomX,out[-1].y+ySize,True,chr(428))]
        out += [Atom(math.e,atomX,out[-1].y+ySize,True,'e')]
        out += [Atom(float('inf'),atomX,out[-1].y+ySize,True,'∞')]
        out += [Atom(float('-inf'),atomX,out[-1].y+ySize,True,'-∞')]
        out += [Atom(chr(i+65),atomX,out[-1].y+ySize+(ySize*i),True) for i in range(26)]
        out += [Atom(' ',atomX,out[-1].y+ySize,True,'space')]
        out += [Atom("''",atomX,out[-1].y+ySize,True,'Silent')]
        out += [Atom(Axis.Axis(),atomX,out[-1].y+ySize,True,'Empty')]
        out += [Atom(True,atomX,out[-1].y+ySize,True)]
        out += [Atom(False,atomX,out[-1].y+ySize,True)]
        out += [Atom(None,atomX,out[-1].y+ySize,True,'Null')]

    def generateFunctions(self,atomX): # Generates all pre-built functions from a .txt file
        ySize = 10
        funcStr = open('PreFunctions.txt','r')
        iFunctions = funcStr.read().splitlines()
        funcStr.close()
        out = [Function.genesis(iFunctions[0],atomX,485+ySize)]
        for i in iFunctions[1:]:
            out += [Function.genesis(i,atomX,out[-1].y+ySize)]

TopLevel(width=800,height=800)
