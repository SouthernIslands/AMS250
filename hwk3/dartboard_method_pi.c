#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#define N 1000000000

int main (int argc, char* argv[]){
    long average, i, sum = 0, total = 0;
    int rank, size;
    double start, end, x, y, pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL));
    average = ((long)N/size);

    for (i = rank ; i < average ; i++){
        x=((double)rand()/RAND_MAX);
        y=((double)rand()/RAND_MAX);

        if(x*x+y*y<1.0){
            sum++;
        }
    }

    MPI_Reduce(&sum, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0){
        pi = ((double) 4*total/N );
        printf("PI is %0.5f computing on %d processes \n", pi, size);
    }

    MPI_Finalize();
    return 0;
}
