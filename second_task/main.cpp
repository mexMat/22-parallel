#define _USE_MATH_DEFINES

#include <iostream>
#include <string>
#include <random>
#include <time.h>
#include <cmath>
#include "mpi.h"

#define GOAL 8.0*M_PI/3.0

using namespace std;

double integrate(int rank, int size, int n) {
  double x, y, z, r, theta, sum = 0.0;
  double offset = (2.0/size);
  double left_x = rank * offset;
  double right_x = (rank+1) * offset;
  double volume = M_PI * (right_x - left_x);

  std::random_device rand_dev;
  std::mt19937 gen(rand_dev());
  std::uniform_real_distribution<> distr_r(0, 1);
  std::uniform_real_distribution<> distr_theta(0, 2.0*M_PI);
  std::uniform_real_distribution<> distr_x(left_x, right_x);


  for(int i = 0; i < n; ++i)
  {
    x = distr_x(gen);
    r = distr_r(gen);
    theta = distr_theta(gen);
    y = r * cos(theta);
    z = r * sin(theta);
    sum += sqrt(y*y + z*z);
    std::cout<<x<<" "<<y<<" "<<z<<std::endl;
  }
  
  sum = volume * sum/n;

  std::cout<<"rank "<<rank<<": "<<sum<<std::endl;

  return sum;
}

int main(int argc, char *argv[]){
  int rank, size, n = 10;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /*..
  double start_time = MPI_Wtime();
  double end_time = MPI_Wtime();
  ..*/

 
  double pie = 0.0, piece = integrate(rank, size, n);
  

  MPI_Reduce(&piece, &pie, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    cout<<"result: "<<pie<<endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
    
  return 0;
}
