EXTRA_CFLAGS ?=

all: fast.omp fast.omp_mpi

fast.omp: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -std=c++11 $(EXTRA_CFLAGS) -fopenmp -pthread -O3 fastBPE/main.cc -IfastBPE -o fast.omp

BOOST_BASE = /pscratch/sd/a/andybna/boost_libs/
fast.omp_mpi: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -O3 -std=c++11 $(EXTRA_CFLAGS) fastBPE/main.cc -fopenmp -IfastBPE \
		-DCONFIG_MPI \
		-L$(BOOST_BASE)/lib -I$(BOOST_BASE) \
		-lboost_mpi -lboost_serialization \
		-lboost_container -lboost_graph \
		-o fast.omp_mpi

clean:
	rm fast.*
