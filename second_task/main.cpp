#define _USE_MATH_DEFINES
#include <iostream>
#include <random>
#include <cmath>
#include "mpi.h"
#define N 25
#define TARGET 4.0*M_PI/3.0
int main(int argc, char *argv[]){
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double y, z, volume;
  volume =  2.0*2.0*2.0/size;
  //define random
  std::random_device rand_dev;
  std::mt19937 gen(rand_dev());
  std::uniform_real_distribution<> dis_y(-1, 1);
  std::uniform_real_distribution<> dis_z(-1, 1);
  double tol = 1.0e-04, eps = 0.0, buf = 0.0, acc_volume, start_time, proc_sum = 0.0, proc_volume = 0.0;
  int iter = 1, stop = 1;

  start_time = MPI_Wtime();
  while(stop)
  {
      for(int i = 0; i < N; ++i)
      {
          y = dis_y(gen);
          z = dis_z(gen);
          if (y*y + z*z <= 1)
              proc_sum += sqrt(y*y + z*z);
      }
      proc_sum = volume * proc_sum;
      MPI_Allreduce(&proc_sum, &proc_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      acc_volume += proc_volume;
      eps = std::abs(TARGET - acc_volume/(iter*N));
      if (eps < tol) {
          stop = 0;
          acc_volume = eps + TARGET;
      }
      proc_sum = 0.0;
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
