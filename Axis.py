# Axis datatype, similar to Python tuples but with several fancy methods and the ability to slice wrap-around

import random

class Axis(object): # Custom immutable list class with some useful methods
    def __init__(self,*values):
        super(Axis, self).__setattr__('values', values) # Initializes values, then freezes them
        super(Axis, self).__setattr__('length', len(values))
    
    __slots__ = ['values','length']
    
    def __setattr__(cls, name, value): # Immutable after initialization
        raise AttributeError(f"Cannot modify Axis {name}")

    def __delattr__(cls, name): # Immutable after initialization
        raise AttributeError(f"Cannot delete Axis {name}")

    def __repr__(self): # Represent an Axis as a list
        if len(self) == 0:
            return 'ø' # Axis(): the empty axis
        if len(self) == 1 and type(self[0]) == Axis: # If the only item is an Axis
            out = '|'                                # use extra | to indicate nesting
            out += str(self[0])
            out += '|'
            return out
        out = '|* '
        for i in range(len(self)):
            nextVal = self[i]
            out += str(self[i])
            if i < len(self)-1:
                out += ' • '
        out += ' *|'
        return out
        
    def __len__(self): # Get the length of an axis
        return self.length

    def __getitem__(self, key): # Index into an Axis (allows wrapping with indices or slices)
        if len(self) == 0:
            return self
        if type(key) == slice:
            if key.step != None and key.step < 0:
                return self[key.stop:key.start:-key.step].reverse()
            if key.start != None and key.start < 0:
                return Axis(*Axis(*self.values,*self.values)[key.start+len(self):key.stop+len(self):key.step])
            if key.stop != None and key.stop >= len(self):
                return Axis(*Axis(*self.values,*self.values)[key])
            return Axis(*self.values[key])
        return self.values[key % len(self)]

    def __setitem__(self, key, value): # Cannot change values of an Axis
        raise TypeError("'Axis' does not support item assignment")
    
    def __delitem__(self, key): # Cannot remove an item from an Axis
        raise TypeError("'Axis' does not support item assignment")

    def __iter__(self): # Turn an Axis into an iterable form
        return iter(self.values)

    def __hash__(self): # Enables the use of Axis in sets & dictionary keys
        return hash(self.values)

    def __eq__(self,other): # Preforms == on Axes and other types based on vaues
        if isinstance(other,Axis):
            return self.values == other.values
        return self.values == other

    def __lt__(self,other): # Preofrms < on Axes and other types using list logic
        if isinstance(other,(list,tuple,dict,set,Axis)):
            return self.values < other
        else:
            return False

    def __le__(self,other): # Preforms <= on Axes and other types using list logic
        if isinstance(other,(list,tuple,dict,set,Axis)):
            return self.values <= other
        else:
            return False

    def __gt__(self,other): # Preforms > on Axes and other types using list logic
        if isinstance(other,(list,tuple,dict,set,Axis)):
            return self.values > other
        else:
            return True

    def __ge__(self,other): # Preforms >= on Axes and other types using list logic
        if isinstance(other,(list,tuple,dict,set,Axis)):
            return self.values >= other
        else:
            return True

    def __add__(self,other): # Adds something to an Axis. If a function is provided,
        if type(other) == type(lambda x: x): # it is used to combine or map the terms
            lambdaVars = len(other.__code__.co_varnames)
            if lambdaVars == 2:
                return self.combine(other)
            if lambdaVars == 1:
                return self.map(other)
            else:
                raise TypeError(f"Cannot '+' with function with {lambdaVars} inputs")
        elif isinstance(other,Axis):
            return Axis(*self,*other)
        elif isinstance(other,(list,tuple,dict,set)):
            if len(other) == 1:
                return Axis(*self,other[0])
            if len(other) == 0:
                return self
            return Axis(*self,*other)
        else:
            return Axis(*self.values,other)

    def __radd__(self,other): # Either append an Axis to another iterable, or incept an atom to an Axis
        if type(other) == list:
            return other + list(self)
        if type(other) == set:
            return other.union(set(self))
        if type(other) == tuple:
            return other + self.values
        if type(other) == str: # Adding an Axis to a string returns a CSV
            return other + ',' + ','.join(map(lambda x: str(x),self))
        if type(other) == type(lambda x: 0): # Adding an Axis to a function applies the function
            return self + other
        return self.incept(other)

    def __mul__(self,other): # How to multiply an Axis
        if isinstance(other,(int,float)): # If Axis * positive int, copy it
            if other == 0:                # If Axis * positive float, slice
                return Axis()             # If Axis * negative, reverse
            if other < 0:
                return self.reverse() * (-other)
            if other == 1:
                return self
            if other < 1:
                return self.take(int(other*len(self)))
            return self + (self * (other - 1))
        if not isinstance(other,(list,tuple,Axis)):
            raise TypeError(f"Cannot '*' Axis with {type(other)}")
            return self
        out = Axis()
        for i in self: # If Axis * iterable, cartesian product
            row = Axis()
            for j in other:
                row = row.append(Axis(i,j))
            out = out.append(row)
        return out

    def __rmul__(self,other): # Apply multiplication commutively
        return self * other

    def __sub__(self,other): # Substracts items from an Axis
        if isinstance(other,(list,tuple,set,Axis)):
            return self.filter(lambda x: x not in other)
        return self.remove(other)

    def __mod__(self,other):
        if isinstance(other,(int)):
            out = Axis()
            for i in range(len(self)):
                if i % other == 0:
                    out = out.append(self[i])
            return out
        return self

    def __div__(self,other):
        print(self,other)
        if isinstance(other,(int,float)):
            return self * (1/other)
        return self

    def floordiv(self,other):
        print(self,other)
        return self.__div__(other)

    def truediv(self,other):
        print(self,other)
        return self.__div__(other)

    def __bool__(self): # An Axis evaluates to True if there is a single nonzero item when flattened
        for i in self.flatten():
            if i != 0:
                return True
        return False

    def column(self,c):
        return self.invert()[c]

    @staticmethod
    def stringSplit(s,par=None): # Returns an Axis that splits up a string
        if par == None:
            return Axis(*list(s))
        print(s,par)
        return Axis(*s.split(par))

    def reverse(self): # Reverses an Axis
        return Axis(*self.values[::-1])

    def reverb(self): # Rexturns list where each value is reversed
        return Axis(*[i[::-1] if isinstance(i,(list,tuple,Axis)) else (i.values[::-1] if isinstance(i,Axis) else i) for i in self.values])

    def invert(self): # Swaps rows and columns
        maxCols = 0
        for row in self:
            if not isinstance(row,(list,tuple,Axis)):
                continue
            maxCols = max(maxCols,len(row))
        out = []
        for c in range(maxCols):
            col = []
            for r in range(len(self)):
                if not isinstance(self[r],(list,tuple,Axis)):
                    if c == 0:
                        col.append(self[r])
                    else:
                        col.append(None)
                elif len(self[r]) > c:
                    col.append(self[r][c])
                else:
                    col.append(None)
            out.append(col)
        if len(out) == 0:
            out = [Axis()]
            for i in self:
                out[0] = out[0].append(i)
        return Axis(*out)

    def rotate(self,n): # Rotates a list counter-clockwise n times
        if n % 4 < 1:
            return self
        out = self.invert().reverse()
        return Axis(*out).rotate(n-1)

    def append(self,value):
        return Axis(*self,value)

    def incept(self,value): # Appends item to the begining
        return Axis(value,*self)

    def head(self): # Returns first item
        if len(self) == 0:
            raise IndexError('Axis index out of range')
        return self[0]

    def tail(self): # Returns all but first item
        if len(self) == 0:
            raise IndexError('Axis index out of range')
        return Axis(*self[1:])

    def last(self): # Returns last item
        if len(self) == 0:
            raise IndexError('Axis index out of range')
        return self[-1]

    def blast(self): # Returns all but last item
        if len(self) == 0:
            raise IndexError('Axis index out of range')
        return Axis(*self[:-1])

    def take(self,n): # Returns first n-1 items (no wraparound)
        return Axis(*self.values[:n])

    def drop(self,n): # Returns items after n (no wraparound)
        return Axis(*self.values[n:])

    def pop(self,n): # Returns all items except item n
        if n in range(len(self)):
            return Axis(*self.take(n) + self.drop(n+1))

    def remove(self,n): # Returns all items except n
        t = -1
        for i in range(len(self)):
            if self[i] == n:
                t = i
                break
        if t != -1:
            return self.pop(t)

    def print(self): # Print out each item
        if len(self) == 0:
            print(self)
        else:
            for i in self:
                print(i)

    def deepSum(self): # Print sum of all integer values (ignores nesting)
        out = 0
        for i in self:
            if isinstance(i,(list,tuple,Axis)):
                out += Axis(*i).deepSum()
            elif isinstance(i,int):
                out += i
        return out

    def flatten(self): # De-nest the axis
        out = []
        for i in self:
            if isinstance(i,(list,tuple,Axis)):
                out += Axis(*i).flatten()
            else:
                out.append(i)
        return Axis(*out)

    def sort(self,reverse=False): # Sort the axis by type or value if same type
        return Axis(*sorted(self,reverse=reverse))

    def purge(self): # Remove duplicates
        out = Axis()
        for i in self:
            if i not in out:
                out = out.append(i)
        return out

    def removeAll(self,n): # Removes all instances of n from an Axis
        return Axis(*filter(lambda x: x != n, self))

    def permutations(self): # Return all permuations of values of any size
        if len(self) == 0:
            return Axis(Axis())
        out = Axis()
        recurse = self.tail().permutations()
        for branch in recurse:
            for i in range(len(branch)+1):
                out = out.append(branch[:i]+self.head()+branch[i:])
        return out

    def powerset(self): # Return the powerset of an Axis (filter with length for combinations)
        if len(self) == 0:
            return Axis(Axis())
        out = Axis()
        recurse = self.tail().powerset()
        for branch in recurse:
            out = out.append(branch)
            out = out.append(self.head() + branch)
        return out

    def rectangular(self): # Does each item of the axis have the same number of items
        for i in self:
            if not isinstance(i,(list,tuple,Axis)) or len(i) != len(self[0]):
                return False
        return True

    def matrix(self,other): # Preforms matrix multiplication
        anti = Axis(*other)
        if not self.rectangular() or not other.rectangular():
            raise ArithmeticError('Cannot preform matrix multiplication on non-rectangular Axis')
        if len(self[0]) != len(other):
            raise ArithmeticError('Axis columns must equal Axis rows to preform matrix multiplicaiton')
        out = Axis(*[[0 for i in range(len(other[0]))] for j in range(len(self))])
        for i  in range(len(self)):
            for j in range(len(other[0])):
                for k in range(len(other)):
                    out[i][j] += self[i][k] * other[k][j]
        return out

    def find(self,item): # Finds indices of all instances an item in an Axis
        out = []
        for i in range(len(self)):
            if self[i] == item:
                out.append(i)
        return Axis(*out)[0]

    def index(self,item): # Finds the index of an item in a list
        for i in range(len(self)):
            if item == self[i]:
                return i
        return None

    def count(self,item): # Counts the number of items that equal a certain item
        out = 0
        for i in self:
            if i == item:
                out += 1
        return out

    def mean(self): # Returns the average of the integers in an Axis
        nums = list(filter(lambda x: isinstance(x,(int,float)), self.flatten()))
        if len(nums) == 0:
            return
            #raise TypeError('mean() of empty sequence with no initial value')
        return sum(nums) / len(nums)

    def median(self): # Returns median element of an Axis
        nums = self.sort()
        if len(nums) == 0:
            return
        m1 = nums[len(nums)//2]
        m2 = nums[(len(nums)-1)//2]
        if not isinstance(m1,(int,float)) or not isinstance(m2,(int,float)):
            return m1
        return (m1 + m2) / 2

    def mode(self): # Returns an Axis of the most frequent items in an Axis
        frequency = 0
        out = Axis()
        for i in self.purge():
            count = self.count(lambda x: x == i)
            if count == frequency:
                out.append(i)
            if count > frequency:
                frequency = count
                out = Axis(i)
        return out

    def map(self,f): # Map f over the items of an Axis
        return Axis(*map(lambda x: f(x),self))

    def filter(self,f): # Filter the items of an Axis with f
        return Axis(*filter(lambda x: f(x),self))

    def combine(self,f=None): # Combine the items of an Axis with f
        if len(self) == 0:
            return Axis()
        if len(self) == 1:
            return self[0]
        out = self[0]
        if f == None: f = lambda x,y: x+y
        else:
            lambdaVars = len(f.__code__.co_varnames)
            if lambdaVars != 2:
                raise TypeError(f'combine() with function with {lambdaVars} inputs')
        for i in self[1:]:
            out = f(out,i)
        return out

    def inject(self,k,n): # Inserts n at index k
        return Axis(*self.take(k),n,*self.drop(k))

    def split(self,k): # Splits Axis at index k
        return Axis(Axis(*self.take(k)), Axis(*self.drop(k)))

    def mutable(self): # Replaces all Axes (and tuples) with lists
        out = []
        for i in self:
            if type(i) == Axis:
                out += [i.mutable()]
            elif type(i) in (list,tuple):
                out.append(Axis(*i).mutable())
            else:
                out += [i]
        return out

    def replace(self,k,n): # Replaces item k with n:
        if n == None:
            return self
        return self.pop(k).inject(k,n)

A0 = Axis()
A1 = Axis(1,2,3,4)
A2 = Axis('a','c','e','g')
A3 = Axis(2,3,5,7,11)
A4 = Axis('x','y','z')
A5 = Axis(A1,A2,A3,A4)
A6 = Axis(0,1,2)
