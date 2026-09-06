CXX = mpicxx
CXXFLAGS = -O2 -Wall -Wextra -std=c++11

all: matmul gen_matrix

matmul: matmul.cpp
	$(CXX) $(CXXFLAGS) -o matmul matmul.cpp

gen_matrix: gen_matrix.cpp
	$(CXX) $(CXXFLAGS) -o gen_matrix gen_matrix.cpp

clean:
	rm -f matmul gen_matrix

.PHONY: all clean
