CFLAGS = -g --std=c++11 `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
OBJS = train.o bow.o histogram.o
DEPS = bow.hpp histogram.hpp
OPT = -O1

%.o: %.cpp $(DEPS)
	g++ -c -o $@ $< $(OPT) $(CFLAGS) $(LIBS)

train: $(OBJS)
	g++ -o $@ $^ $(OPT) $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm train *.o
