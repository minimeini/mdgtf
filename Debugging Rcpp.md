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



## Setting up Rcpp Compiler for Apple Silicon

1. Follow [R COMPILER TOOLS FOR RCPP ON MACOS](https://thecoatlessprofessor.com/programming/cpp/r-compiler-tools-for-rcpp-on-macos/)
2. Regarding the `gfortran` part, do NOT follow the first link. Instead, check out [Tools - R for Mac OS X](https://mac.r-project.org/tools/)
    - Most importantly, please install the mandatory libraries via the [recipes system](https://mac.r-project.org/tools/) so that R can compile source codes.

## Sanity Check

Output of `xcode-select -v`:

```bash
xcode-select version 2396.
```

Output of `xcode-select -p`:

```bash
/Library/Developer/CommandLineTools
```

Output of `gcc --version`:

```bash
Apple clang version 14.0.0 (clang-1400.0.29.202)
Target: arm64-apple-darwin22.2.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```

Output of `R.version.string` in `R`:

```r
> R.version.string
[1] "R version 4.2.2 (2022-10-31)"
```

My `macOS` version:

```bash
System Version: macOS 13.1 (22C65)
Kernel Version: Darwin 22.2.0
Processor: Apple M1 Pro
```

Locations of R-related headers:

```r
> R.home()
[1] "/Library/Frameworks/R.framework/Resources"
> RcppArmadillo:::CxxFlags()
-I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/library/RcppArmadillo/include"
```

Location of R packages

```r
> .libPaths()
"/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/library"
```