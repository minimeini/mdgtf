# Debugging Rcpp

The steps:

1) (One off) Go to the directory ~/.R (a hidden directory with the .). Create a new file called "Makevars" and in it add the line CXXFLAGS=-g -O0 -Wall.

2) In the terminal, type R -d lldb to launch R. lldb will now start.

3) Type run at the lldb command line. This will start R.

4) Compile the Rcpp code and find the location of the compiled objects. Dirk's response to the above mentioned post shows one way to do this. Here is an example I'll use here. Run the following commands in R:

library(inline)

fun <- cxxfunction(signature(), plugin="Rcpp", verbose=TRUE, body='
int theAnswer = 1;
int theAnswer2 = 2;
int theAnswer3 = 3;
double theAnswer4 = 4.5;
return wrap(theAnswer4);
')
This creates a compiled shared object and other files which can be found by running setwd(tempdir()) and list.files() in R. There will be a .cpp file, like "file5156292c0b48.cpp" and .so file like "file5156292c0b48.so"

5) Load the shared object into R by running dyn.load("file5156292c0b48.so") at the R command line

6) Now we want to debug the C++ code in this .so object. Go back to lldb by hitting ctrl + c. Now I want to set a break point at a specific line in the file5156292c0b48.cpp file. I find the correct line number by opening another terminal and looking at the line number of interest in file5156292c0b48.cpp. Say it's line 31, which corresponds to the line int theAnswer = 1; above in my example. I then type at the lldb command line: breakpoint set -f file5156292c0b48.cpp -l 31. The debugger prints back that a break point has been set and some other stuff...

7) Go back to R by running cont in lldb (the R prompt doesn't automatically appear for me until I hit enter) and call the function. Run fun() at the R command line. Now I am debugging the shared object (hit n to go to next line, p [object name] to print variables etc)....