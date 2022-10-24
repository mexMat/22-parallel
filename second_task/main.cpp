#define _USE_MATH_DEFINES
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include "mpi.h"
#define TARGET 4.0*M_PI/3.0
int main(int argc, char *argv[]){
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double y, z, volume = 8.0/size;
  int N = 100;
  //define random
  double tol = std::atof(argv[1]), eps = 0.0, acc_volume = 0.0, start_time, proc_sum = 0.0;
  int iter = 1, stop = 1;

  start_time = MPI_Wtime();
  srand(rank+1);
  while(stop)
  {
    for(int i = 0; i < N; ++i)
    {
      y = -1 + 2.0*(double)rand()/(RAND_MAX+1.0);
      z = -1 + 2.0*(double)rand()/(RAND_MAX+1.0);
      if (y*y + z*z <= 1)
        proc_sum += sqrt(y*y + z*z);
    }
    MPI_Allreduce(&proc_sum, &acc_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    eps = std::abs(TARGET - volume*acc_volume/(iter*N));
    if (eps < tol)
    {
      stop = 0;
      acc_volume = TARGET + eps;
    }
    iter++;
  }
  double end_time = MPI_Wtime();
  double result_time, time = end_time - start_time;
  MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout<<"True: "<<TARGET<<std::endl;
    std::cout<<"Integral: "<<acc_volume<<std::endl;
    std::cout<<"Eps: "<<eps<<std::endl;
    std::cout<<"N points: "<<size * N * iter<<std::endl;
    std::cout<<"Time: "<<result_time<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
