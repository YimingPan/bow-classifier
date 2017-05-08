CFLAGS = -g --std=c++11 `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
OBJS = train.o bow.o histogram.o
DEPS = bow.hpp histogram.hpp
OPT = -O2
OMPFLAGS = -fopenmp

%.o: %.cpp $(DEPS)
	g++ -c -o $@ $< $(OPT) $(OMPFLAGS) $(CFLAGS) $(LIBS)

train: $(OBJS)
	g++ -o $@ $^ $(OMPFLAGS) $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm train *.o
