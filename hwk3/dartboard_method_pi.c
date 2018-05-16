#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#define N 1E8
#define d 1E-8

int main (int argc, char* argv[]){
    int rank, size, result=0, sum=0;
    double start, end, x, y;
    
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    srand((int)time(0));
    
    for (i=rank; i<N; i+=size){
        x=rand()/(RAND_MAX+1.0);
        y=rand()/(RAND_MAX+1.0);
        if(x*x+y*y<1.0)
            result++;
    }
    
    //Sum up all results
    MPI_Reduce(&result, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    //Synchronize all processes and get the end time
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    //Caculate and print PI
    if (rank==0){
        pi=4*d*sum;
        printf("np=%2d;    Time=%fs;    PI=%0.4f\n", size, end-begin, pi);
    }
    
    MPI_Finalize();
    return 0;
}
