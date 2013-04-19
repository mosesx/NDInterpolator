#======================================================================
# USER SETTINGS
#======================================================================

# Path to boost (inlcudes)
BOOSTPATH = /home/mrehberg/extPrograms/boost
BOOSTBINDINGS= /home/mrehberg/extPrograms/boost_numeric_bindings
NDINTERPOLATOR= /home/mrehberg/projekte/NDInterpolator/r25
# Path to serialization library
SERIALLIB_PATH = /home/mrehberg/extPrograms/boost/stage/lib
# Path to CPPAD
CPPAD_PATH=/home/mrehberg/extPrograms/cppad

# optimization, debug and warning flags for the compiler
OFLAGS= -march=native -O3 
DFLAGS= -DNDEBUG
WFLAGS= -Wall -pedantic -w

# serialization support enabled  
WO_SERIAL=FALSE

# sourcen, optShape also demonstrates serialization --> do not compile if serialization support is
# not enabled
SRC=example.cpp \
	example2.cpp\
	optShape.cpp \
	CppADExample.cpp \
	testBump.cpp \
	sinExampleOptShapePUH.cpp


SRC_WO_S=example.cpp \
	example2.cpp \
	CppADExample.cpp 	

#==========================================================================


INCL = -I$(BOOSTPATH) -I$(BOOSTBINDINGS) -I$(NDINTERPOLATOR) -I$(CPPAD_PATH) -I./

ifeq ($(WO_SERIAL),TRUE)   
	LIBS = -llapack
	LIB_PATH=
	SRC=$(SRC_WO_S)	
else
	LIB_PATH=-L$(SERIALLIB_PATH)
	LIBS= -llapack -lboost_serialization -lboost_system -lboost_timer
endif

EXEC=$(SRC:.cpp=)
OBJS=$(SRC:.cpp=.o)

CC = g++


all: $(EXEC)

$(EXEC): % : %.o
	$(CC) $(LIB_PATH) $(LIBS) $(OFLAGS) $(DFLAGS) $(WFLAGS) -std=c++0x  -o $@ $< 

$(OBJS): %.o : %.cpp
	$(CC) $(INCL) $(OFLAGS) $(DFLAGS) $(WFLAGS) -std=c++0x -c $<

clean:
	rm -f $(EXEC) *.o *.dat

.PHONY: clean