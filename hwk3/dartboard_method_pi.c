#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#define N 10000000

int main (int argc, char* argv[]){
    int i, sum = 0, total = 0;
    long rank, size, average;
    double start, end, x, y, pi=0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    average = N/size;
    srand((int)time(123));

    for (i = rank ; i < average ; i++){
        x=rand()/(RAND_MAX+1.0);
        y=rand()/(RAND_MAX+1.0);

        if(x*x+y*y<1.0){
           sum++;
        }
    }

    //Sum up all results
    MPI_Reduce(&sum, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    //Caculate and print PI
    if (rank == 0){
        pi = (double) 4*total/N;
        printf("np=%2d   PI=%0.5f\n", size, pi);
    }

    MPI_Finalize();
    return 0;
}
