CFLAGS = -g --std=c++11 `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
OBJS = bow.o histogram.o
DEPS = bow.hpp histogram.hpp
OPT = -O2
OMPFLAGS = -fopenmp

all: train evaluate

%.o: %.cpp $(DEPS)
	g++ -c -o $@ $< $(OPT) $(OMPFLAGS) $(CFLAGS)

train: train.o $(OBJS)
	g++ -o $@ $^ $(OMPFLAGS) $(CFLAGS) $(LIBS)

evaluate: evaluate.o $(OBJS)
	g++ -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm train evaluate *.o
