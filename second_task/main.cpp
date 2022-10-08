#define _USE_MATH_DEFINES
#include <iostream>
#include <random>
#include <cmath>
#include "mpi.h"
#define N 100
#define TARGET 4.0*M_PI/3.0
int main(int argc, char *argv[]){
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double x, y, z, volume;
  volume =  2.0*2.0*2.0/size;
  //define random
  std::random_device rand_dev;
  std::mt19937 gen(rand_dev());
  std::uniform_real_distribution<> dis_y(-1, 1);
  std::uniform_real_distribution<> dis_z(-1, 1);
  double tol = 1.0e-05, eps = 0.0, buf = 0.0, acc_volume, start_time, proc_sum = 0.0, proc_volume = 0.0;
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
      proc_sum = volume * proc_sum/(iter*N);
      MPI_Reduce(&proc_sum, &proc_volume, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rank == 0)
      {
          acc_volume = (iter-1)*acc_volume/iter + proc_volume;
          eps = std::abs(TARGET - acc_volume);
          if (eps < tol)
              stop = 0;
      }
      ++iter;
      MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  double end_time = MPI_Wtime();
  double result_time, time = end_time - start_time;
  MPI_Reduce(&time, &result_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout<<"True: "<<TARGET<<std::endl;
    std::cout<<"Integral: "<<acc_volume<<std::endl;
    std::cout<<"Eps: "<<std::abs(TARGET - acc_volume)<<std::endl;
    std::cout<<"N points: "<<size * N * iter<<std::endl;
    std::cout<<"Time: "<<result_time<<std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
