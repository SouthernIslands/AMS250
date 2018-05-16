#include <time.h>
#include "mpi.h"
#include <stdio.h>

int main(int argc,char *argv[]){
    int i;
    int size, rank, tag=1;
    double start, end, time;
    int *msgsend = (int *) malloc(10000*sizeof(int));
    int *msgrecv = (int *) malloc(10000*sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){

      start = MPI_Wtime();
      for(i = 1; i <= 100 ; i++){
        MPI_Send(&msgsend[0], 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
        MPI_Recv(&msgrecv[0], 1, MPI_INT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      end = MPI_Wtime();
      time = end - start;

      printf("Average time is %f seconds\n", time/100);
    }
    if(rank == 1){
      for(i = 1; i <= 100 ; i++){
        MPI_Recv(&msgrecv[0], 1, MPI_INT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&msgsend[0], 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
      }
    }

}
