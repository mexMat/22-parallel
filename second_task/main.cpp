#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include "mpi.h"
#define TARGET 4.0*M_PI/3.0

int main(int argc, char *argv[]) {
    int i, rank, size, iter = 1, stop = 1, block_size = 20;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double r, proc_sum = 0.0, acc_sum = 0.0, result = 0.0, eps = std::atof(argv[1]);
    srand(rank + 1);
    double start_time = MPI_Wtime();
    while(stop){
        for(i = 0; i < block_size; ++i)
        {
            r = (double)rand()/RAND_MAX;
            proc_sum += r*r;
        }
        MPI_Allreduce(&proc_sum, &acc_sum, 1, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);
        result = 4.0*M_PI*acc_sum/(iter*size*block_size);
        if (std::fabs(TARGET - result) < eps){
            stop = 0;
        }
        ++iter;
    }
    double end_time = MPI_Wtime();
    double result_time, time = end_time - start_time;
    MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "True: " << TARGET << std::endl;
        std::cout << "Integral: " << result << std::endl;
        std::cout << "Eps: " << std::abs(TARGET - result) << std::endl;
        std::cout << "N points: " << size * block_size * iter << std::endl;
        std::cout << "Time: " << result_time << std::endl;
    }
    MPI_Finalize();
    return 0;
}