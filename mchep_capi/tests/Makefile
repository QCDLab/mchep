CC = gcc
CXX = c++ -std=c++11
CFLAGS = -O3 -Wall -Wextra -Werror
CXXFLAGS = -O3 -Wall -Wextra -Werror
MCHEP_DEPS != pkg-config --cflags --libs mchep_capi
MATH_LIBS = -lm

PROGRAMS = test_capi test_cppapi

all: $(PROGRAMS)

test-examples: $(PROGRAMS)
	set -e && for i in $(PROGRAMS); do ./$${i} > output; diff -u $${i}.output output; done; rm output

test_capi: test_capi.c
	$(CC) $(CFLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

test_cppapi: test_cppapi.cpp
	$(CXX) $(CXXFLAGS) $< $(MCHEP_DEPS) $(MATH_LIBS) -o $@

.PHONY: clean

clean:
	rm -f $(PROGRAMS) *.neopdf.lz4
