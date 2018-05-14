#include "mpi.h"
#include <stdio.h>

int main(int argc,char *argv[]){
    int message;
    int size, rank, prev, next, tag;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    tag = 43;
    next = (rank + 1) % size;
    prev = (rank + size - 1) % size;


    if (rank == 0) {
        message = 16;
        
        printf("Master process (Process 0) initating message %d send to %d of %d Process ring\n", 
        message, next, size);
        MPI_Send(&message, 1, MPI_INT, next, tag, MPI_COMM_WORLD); 
    }


    while (1) {
        MPI_Recv(&message, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, 
            MPI_STATUS_IGNORE);
        printf("Process %d received message %d from process %d\n",
        rank, message, prev);

        if (rank == 0){
            message--;
            printf("-----------------------------------------\n");
            printf("Message decrese to %d at master process\n", message);
            printf("-----------------------------------------\n");
        }
        

        printf("Process %d sended message %d to process %d\n",
        rank, message, next);
        MPI_Send(&message, 1, MPI_INT, next, tag, MPI_COMM_WORLD);

        if (message == 0) {
           
            printf("Process %d quitting the ring\n", rank);
            break;
    }
  }
        
  /*  Careful! Final receive before program ends */

    if (rank == 0) {
            MPI_Recv(&message, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, 
                MPI_STATUS_IGNORE);
            printf("****Final receive before program ends****\n");
    }

    MPI_Finalize();
    return 0;

}
