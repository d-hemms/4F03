PROGRAM_NAME=assig1.x
CFLAGS=-g -Wall -std=c++11 -fopenmp
CC=g++

$(PROGRAM_NAME):	main.cpp f.cpp
	$(CC) -O2 -o $@ $^ $(CFLAGS)

clean:
	rm assig1.x
