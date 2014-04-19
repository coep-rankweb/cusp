#include <cusp/io/matrix_market.h>
#include <cusp/print.h>
#include <cusp/verify.h>
#include <cusp/coo_matrix.h>
#include <cusp/array1d.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/detail/timer.h>

#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

#define THRESHOLD 0.000001f

#ifdef CPU

#define CAST(X)	X ## _CPU
#define MEM	cusp::host_memory

#else

#define CAST(X)	X ## _GPU
#define MEM	cusp::device_memory

#endif

#define ABS(x)	((x) < 0 ? -(x) : (x))

typedef cusp::coo_matrix<int, double, cusp::host_memory> COO_CPU;
typedef cusp::array1d <double, cusp::host_memory> ARR1D_CPU;

typedef cusp::coo_matrix<int, double, cusp::device_memory> COO_GPU;
typedef cusp::array1d <double, cusp::device_memory> ARR1D_GPU;

/*
	Creating a struct to define the inversion operation. 
*/
template <typename T>
struct inversion_op {
__host__ __device__ T operator()(const T& x)const{
		return ABS(x) > 0 ? 1 / x : 0;
}
};

/*
	Creating a struct to define an operation to detect dangling nodes 
*/
template <typename T>
struct dangling_op {
__host__ __device__ T operator()(const T& x)const{
		return x > 0 ? 0 : 1;
}
};

/*
	Creating a struct to count the number of nodes above THRESHOLD	
*/
template <typename T>
struct check_above_threshold {
__host__ __device__ T operator()(const T& x)const{
		return x > THRESHOLD ? 1 : 0;
}
};

extern void read_matrix(CAST(COO) &temp, const char *fname);

extern void normalize(CAST(COO) &adj, CAST(ARR1D) &dangling);

extern void print_array(CAST(ARR1D) rank);

extern void pagerank(CAST(COO) &link, double beta, CAST(ARR1D) &rank, CAST(ARR1D) &dangling);
