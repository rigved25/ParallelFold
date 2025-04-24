CC=g++
CFLAGS=-std=c++14

all: rna rnamp rnal2r

rna: rna.cpp
	mkdir -p bin
	$(CC) $(CFLAGS) rna.cpp -o ./bin/rna

rnamp: rna_mp.cpp
	mkdir -p bin
	$(CC) $(CFLAGS) -fopenmp rna_mp.cpp -o ./bin/rna_mp

rnal2r: rna_l2r.cpp
	mkdir -p bin
	$(CC) $(CFLAGS) -fopenmp rna_l2r.cpp -o ./bin/rna_l2r

clean:
	rm -f ./bin/rna ./bin/rna_mp ./bin/rna_l2r
