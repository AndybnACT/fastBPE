

all: omp_tbb omp_merge omp_critical mpi

omp_tbb: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -O3 -std=c++11  fastBPE/main.cc -fopenmp -IfastBPE -DCONFIG_OMP -DCONFIG_OMP_TBB -tbb -o fast.omp_tbb

omp_merge: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -O3 -std=c++11  fastBPE/main.cc -fopenmp -IfastBPE -DCONFIG_OMP -DCONFIG_OMP_SINGLE_THREADED_MERGE -o fast.omp_stm

omp_critical: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -O3 -std=c++11  fastBPE/main.cc -fopenmp -IfastBPE -DCONFIG_OMP -DCONFIG_OMP_CRITICAL -o fast.omp_crt

BOOST_BASE = /pscratch/sd/a/andybna/boost_libs/
mpi: fastBPE/main.cc fastBPE/fastBPE.hpp
	CC -O3 -std=c++11  fastBPE/main.cc -fopenmp -IfastBPE \
		-L$(BOOST_BASE)/lib -I$(BOOST_BASE) \
		-lboost_mpi -lboost_serialization \
		-lboost_container -lboost_graph \
		-o fast.mpi

clean:
	rm fast.*
