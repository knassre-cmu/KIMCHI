Kian Nassre
15-112 Design Proposal

Project Description: KIMCHI (Kian’s Incredible Meta Computer Human Interaction)
	<I had to name it something, so why not come up with an adorable acronym?>

A drag-and-drop programming language, similar to Scratch or Snap!, but geared more heavily towards evaluation rather than statements (i.e., conforming to a functional paradigm rather than a procedural or imperative paradigm). There will be 3 major modes: a variable “sandbox” where functions and atoms (fundamental values like 7, A, False, None) can be dragged in and out of each other and into a set of slots representing (limited) variables, which can be evaluated; a function creator where a new function can be defined with up to 9 inputs by composing other functions and atoms; and an animation sandbox where pre-defined animation functions can be used like drawing instructions and take functions, atoms, or variables as inputs, then run to produce an animation.

Competitive Analysis: Scratch, and it’s less famous cousin Snap! (which is slightly more similar to my project) are well established tools for getting individuals acquainted with computer science in a fun atmosphere without having to dredge them in coding. While they have a rather heavy emphasis on the animation side (since it is the most kid-friendly), mine will have a much greater emphasis on the functional paradigm of coding, since a great limitation of my language will be the fact that almost everything is an expression rather than a statement. Even though some of my pre-built functions use loops and other frowned upon things in functional programming, the fact that they can only be used in an evaluative form. As a result, mine still serves as a psychological warm up to functional side of programming, which can be a jolt for those who are more familiar with the procedural and imperative, since the flow of data between functions is a more abstract concept than a statement saying DO THIS or an object storing some data.

In terms of graphics, there are key similarities and differences. Like Snap! And Scratch, mine will have a library of functions that can be used on the left hand side, color coordinated based on type (string functions are green, math functions are blue, predicate functions are purple, etc.). I also share the ‘block’ structure and have a somewhat similar look and feel. However, unlike Scratch or Snap, there is no typing inputs into functions manually. The base components of each composed function must be atoms. Also, unlike Scratch or Snap! which allow blocks to be placed anywhere and called by commands such as “When CLICKED,” and which tie these blocks to the sprite that tis currently being worked on, everything is in one sequential set of variables. Things can be dragged into or out of the variable slots and discarded, but they don’t have a place to just hang around. Also, the animation mode is separate.

Structural Plan: I will use a Modal App with modes for the splash screen, sandbox, creator and animator. There is a superclass for all the blocks: Dragger. Within Dragger there are functions and atoms (there are some noticeable differences in their code since functions act as recursive calls and atoms act as base cases). Within functions there are subclasses for each type to color coordinate and in case I decide to do further modifications to certain types of functions. For instance, at the moment, all function calls are under a try-except. If an input is invalid or empty, it is replaced with a pre-defined default atom. But perhaps I could use the individual subclasses of function to define a more rigorous method of type checking (this could be especially useful for HOFs like map, filter, combine, optimize, minimize, sort and find-first). How much of this main code is in one file vs several files is to be seen.

There are several external files that are used. First is the .txt file storing all the pre-built functions. Upon loading, the main algorithm uses string processing and eval() to create the library of functions based on these, which I can add too as I see fit. Also, when I add a function, I will be able to write too this file or use a similar structure. A similar process will be done for saving & loading the state of the sandbox and animator. At this time, what I am thinking is that the pre-built functions file will remain separate, while each save will be a unique .txt file storing any functions created, the sandbox, and the animator all in one. Also, there is the file Axis.py. Back when we learned magic methods, I began experimenting with some of the stranger ones, and learned how to define my own iterative data type using __getitem_(). Since then, I have tinkered with my own iterable: the Axis. An Axis is similar to a Python Tuple but with some fancy tricks. For instance, I made it so that slicing will wrap around. Also they are immutable (trying to change the values will raise an exception). After much tinkering, I have implemented Axes into my project in lieu of Python lists. I also use the 15-112 graphics file (although I do some unorthodox manipulation of it).

Algorithmic Plan: There are two difficult parts of my project. One of them I have already 95% solved: the slotting in/out of functions/atoms into one another to compose. Although the graphics of the nested functions are still finicky, it does work. Although the implementation was not very difficult, it took a required a lot of iteration in earlier versions of the code in order to prepare the Dragger & Function classes. This required re-writing my code top to bottom on several occasions (copy and pasting large swaths) in order to revise the core implementation of the functions, especially with regards to the difference between the cloneable blocks in the scrollable library and the clones in the sandbox. The result is thus: Dragger has a method .touching() which is called by mousePressed on each clone able and non-cloneable block to see if the dragging point (a red circle that turns green while being dragged) is within 10px of the cursor. If so, a new function/atom is cloned from the original and returned to mousePressed, which will assign it to a variable which controls the movement when dragged. If the thing being pressed is a clone, then it is just moved directly. The key factor is that a method called .slotify() is used at every junction; .slotify() is used to position each child funciton/atom within its parent function after it is placed, and when the parent is moved (a separate method called .killChildren() recursively deletes all nested blocks when a block is dragged into an area that indicates discarding). The mechanic that allows the placement/removal of children and the accordion-ing of the parents is that each Dragger has a parent attribute (None if at the top) and a slots attribute which is a list of all its children (or None if a slot is empty). When a function is called, the __call__() method calls the associated lambda expression (in the .txt file used to generate the library) on each of the operands. If a slot is empty, the default atom is used. If the slot has an atom in it, the atoms value is used. If the slot has a function in it, a recursive call is done. At this stage, all of this works except for a glitch with the graphics of a function with nested functions (the accordion-ing is not entirely correct, but they can still be placed/removed and evaluated).

The next difficult problem, which is the most important part of my project, is the function-creation mode. My intent is that it will work as thus: when the user wishes to create a new function, they will be prompted to enter a name and a number of inputs between 0 and 9, as well as select from a type of function: arithmetic (math), letter (string), iterable (list), predicate (boolean) or control (a category that simulates statements such as conditionals and loops). After doing so, they will enter a mode similar to the variable sandbox but with only 1 slot: the value that will be returned when the new function is called. Additionally, the new function will appear in the library, thus allowing recursion, and the inputs of the function can be used as atoms. When the new function is done, it will be created like the pre-built functions were. If the state is saved, the new function’s name, type and evaluation protocol will be put into a string format similar to the pre-built and saved along with the sandbox & visualizer state in a .txt file which can be loaded. To accomplish this I will first complete the variable sandbox, since the graphic components will be similar. I will then experiment with the translation of a composed function to a string format, since it breaks from the traditional lambda form I use for the pre-built functions. Afterwards, I will focus on the saving of the function in tandem with the variable sandbox (so that folding in the animator later on won’t be a hassle).

Timeline plan: I will essentially have the first difficult problem, the manual composition of functions, done by TP1. I intend to have the variable sandbox, the function creator, and the saving of functions/state done by TP2. Afterwards, I will focus on the animator and interactive perks such as a helper screen and complex demos that can be loaded. Additionally, I will work on improving the quality of graphics as I go along (one of my early versions already has a cute splash screen).

Version Control Plan: I will use my GitHub account to store the old versions that got re-written as well as occasional updates.

Module List: math, random, copy, decimal, time, TKinter, CMU graphics file, Axis, possibly OS, possibly functools, possibly time.