#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#define N 10000000

int main (int argc, char* argv[]){
    int rank, size, sum=0, total=0;
    double start, end, x, y, pi=0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand((int)time(0));

    for (i = rank ; i < N ; i += size){
        x=rand()/(RAND_MAX+1.0);
        y=rand()/(RAND_MAX+1.0);

        if(x*x+y*y<1.0){
           sum++;
        }
    }

    //Sum up all results
    MPI_Reduce(&sum, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    //Caculate and print PI
    if (rank==0){
        pi = (double) 4*total/N;
        printf("np=%2d   PI=%0.5f\n", size, pi);
    }

    MPI_Finalize();
    return 0;
}
