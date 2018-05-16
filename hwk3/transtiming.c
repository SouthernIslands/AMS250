#include "mpi.h"
#include <stdio.h>
#define N 1000
#define L 1000000

int main(int argc,char *argv[]){
    int i, len = 1;
    int size, rank, tag=1;
    double start, end, time;
    int *msgsend = (int *) malloc(L*sizeof(int));
    int *msgrecv = (int *) malloc(L*sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        while(len <= L){
          start = MPI_Wtime();
          for(i = 1; i <= N ; i++){
            MPI_Send(&msgsend[0], len, MPI_INT, 1, tag, MPI_COMM_WORLD);
            MPI_Recv(&msgrecv[0], len, MPI_INT, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
          end = MPI_Wtime();
          time = end - start;

          printf("Average time is %f seconds\n", time/N);
          printf("While message length is %d \n", len);  
        } 
      len = len*100;  
      }  
    }
    if(rank == 1){
      while(len <= N){
      for(i = 1; i <= N ; i++){
        MPI_Recv(&msgrecv[0], len, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&msgsend[0], len, MPI_INT, 0, tag, MPI_COMM_WORLD);
        }
        len = len*100;
      }
    }
    
    MPI_Finalize();
    return 0;
}
