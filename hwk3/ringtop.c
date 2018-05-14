#include "mpi.h"
#include <stdio.h>

int main(int argc,char *argv[]){
    int message;
    int size, rank, prev, next, tag=1;
    MPI_Request reqs[2];
    MPI_Status stats[2];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    tag = 3; 
    next = (rank + 1) % size;
    prev = (rank + size - 1) % size;

    if (rank == 0) {
        message = 16;
        
        printf("Master process (Process 0) initating message %d send to %d of %d Process ring\n", 
        message, next, size);
        MPI_Isend(&message, 1, MPI_INT, next, tag, MPI_COMM_WORLD, &reqs[0]); 
        MPI_Wait(&reqs[0], &stats[0]);
    }

    while (1) {
            MPI_Irecv(&message, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, &reqs[1]);   
            MPI_Wait(&reqs[1], &stats[1]);
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
        MPI_Isend(&message, 1, MPI_INT, next, tag, MPI_COMM_WORLD, &reqs[0]);
        MPI_Wait(&reqs[0], &stats[0]);

        if (message == 0) {
            
                printf("Process %d quitting the ring\n", rank);
                break;
            }
    }

    /*  Careful! Final receive before program ends */
    
    if (rank == 0) {
            MPI_Irecv(&message, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, 
                &reqs[1]);
            MPI_Wait(&reqs[1], &stats[1]);
            printf("****Final receive before program ends****\n");
    }

    MPI_Waitall(2, reqs, stats);
    MPI_Finalize();
    return 0;
}
