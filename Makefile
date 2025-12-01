

all: fast.mpi fast.omp_tbb fast.omp_stm fast.omp_crt fast.orig

fast.orig: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast.orig

fast.omp_tbb: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -O3 -std=c++11  fastBPE/main.cc -fopenmp -IfastBPE -DCONFIG_OMP -DCONFIG_OMP_TBB -tbb -o fast.omp_tbb

fast.omp_stm: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -O3 -std=c++11  fastBPE/main.cc -fopenmp -IfastBPE -DCONFIG_OMP -DCONFIG_OMP_SINGLE_THREADED_MERGE -o fast.omp_stm

fast.omp_crt: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -O3 -std=c++11  fastBPE/main.cc -fopenmp -IfastBPE -DCONFIG_OMP -DCONFIG_OMP_CRITICAL -o fast.omp_crt

BOOST_BASE = /pscratch/sd/a/andybna/boost_libs/
fast.mpi: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -O3 -std=c++11  fastBPE/main.cc -fopenmp -IfastBPE \
		-DCONFIG_MPI \
		-L$(BOOST_BASE)/lib -I$(BOOST_BASE) \
		-lboost_mpi -lboost_serialization \
		-lboost_container -lboost_graph \
		-o fast.mpi

clean:
	rm fast.*
