#include "mpi.h"
#include <stdio.h>
#include <sys/time.h>

#define ROWSIZE 40
#define NUMBERROWS 10

char *DISH0[NUMBERROWS];
char *DISH1[NUMBERROWS];
char *defau[NUMBERROWS] = {
  "***OOOO*OO****OOO****************OO*OO*O",
  "O**OOO**OOO****O******************OO****",
  "OO*OOO****************O**O***O*****O****",
  "**O**O***O************O*****O***********",
  "****OO**************OO*OOO*************O",
  "*O*******OOO*OOO******O***************OO",
  "****OO***O****OOO**OOO*OOO***********O**",
  "O******O**O******O******O***********OO**",
  "*O******OOO******O********************O*",
  "****OOO**OO******O******************O*O*",
};


void cleanScreen(){
  /*
  * brings the cursor home (top-left), so that the next generation will
  * be printed over the current one.
  */

 char ANSI_CLS[] = "\x1b[2J";
 char ANSI_HOME[] = "\x1b[H";
 printf( "%s%s", ANSI_HOME, ANSI_CLS );

}

void initDish(){
  int i;
  for( i = 0; i < NUMBERROWS ; i++ ){
    DISH0[i] = (char *) malloc( (strlen( defau[0] ) + 1 ) * sizeof( char ) );
    strcpy( DISH0[i], defau[i] );

    DISH1[i] = (char *) malloc( (strlen( defau[0] ) + 1 ) * sizeof( char ) );
    strcpy( DISH1[i], defau[i] );
  }
}

void print(char ** dish, int firstRow, int lastRow){
    // char ANSI_HOME[] = "\x1b[H";
    // printf( "%s", ANSI_HOME );
    int i;

    for (i=firstRow; i<=lastRow; i++ ) {
    if ( dish[i] == NULL ) continue;
    pos( i, 0 );
    printf( "%s\n", dish[i] );
    }
}

void  life( char** dish, char** next, int lowerRow, int upperRow ) {
  /*
   * Given an array of string representing the current population of cells
   * in a petri dish, computes the new generation of cells according to
   * the rules of the game. A new array of strings is returned.
   */
  int i, j, k, row;
  int rowLength = (int) strlen(dish[0]);
  int dishLength = NUMBERROWS;

  for (row = lowerRow; row < upperRow; row++) {// each row
    
    if ( dish[row] == NULL )
      continue;

    for ( i = 0; i < rowLength; i++) { // each char in the row

      int r, j, neighbors = 0;
      char current = dish[row][i];

      // loop in a block that is 3x3 around the current cell
      // and count the number of '#' cells.
      for ( r = row - 1; r <= row + 1; r++) {
        // assume periodic boundary conditions
        // make sure we wrap around from bottom to top
        int realr = r;
        if (r == -1)
          realr = dishLength - 1;
        if (r == dishLength)
          realr = 0;

        for ( j = i - 1; j <= i + 1; j++) {

          // make sure we wrap around from left to right
          int realj = j;
          if (j == -1)
            realj = rowLength - 1;
          if (j == rowLength)
            realj = 0;

          if (r == row && j == i)
            continue; // current cell is not its neighbor
          if (dish[realr][realj] == 'O')
            neighbors++;
        }
      }

      if (current == 'O') {
        if (neighbors < 2 || neighbors > 3)
          next[row][i] =  '*';
        else
          next[row][i] = 'O';
      }

      if (current == '*') {
        if (neighbors == 3)
          next[row][i] = 'O';
        else
          next[row][i] = '*';
      }
    }
  }
}

int main(int argc, char* argv[]){
  
  char **dish, **next, **temp;
  int nextProcess, prevProcess;
  int gens = 10;
  int i = 0;
  struct timeval tv;
  double start, stop, timespend;
  int n; // number of tasks/processes
  int sizeOfSection, lastRow, firstRow;


  int rank,noTasks;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&noTasks); 

  if( noTasks % 2 != 0){
    printf( "Number of Processes/Tasks must be even.  Number = %d\n\n", noTasks );
    MPI_Finalize();
    return 1;
  } else {
    n = noTasks; 
  } 
  
  cleanScreen();
  initDish();

  dish = DISH0;
  next = DISH1;

  sizeOfSection = NUMBERROWS/n;
  firstRow =  sizeOfSection * rank; //give rows based on rank
  lastRow = firstRow + sizeOfSection; //assign rows based on rank

  if(rank == 0){
    prevProcess = n - 1;
  } else {
    prevProcess = rank - 1;
  }

  if(rank == n - 1){ // if it's the last process
    nextProcess = 0;
  } else { 
    nextProcess = rank + 1;
  }  


  if(i == 0){
    print(dish, firstRow, lastRow);
    printf("\n\n");
    gettimeofday(&tv,NULL);
    start = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
  }

  for ( i = 0; i < gens; i++) {   
    
    // apply the rules of life to the current population and 
    // generate the next generation.
    life( dish, next, firstRow, lastRow );
    // if(rank == 0){
    //   print(dish);
    //   printf( "Rank %d: Generation %d\n\n", rank, i );
    // }

    //--- if rank is odd, then send ---
    //--- if rank is even, then receive ---
    if(rank % 2 == 0){ //even
    //          buffer              #items  item-size     src/dest tag          world  
      MPI_Send( next[firstRow],   ROWSIZE, MPI_CHAR,    prevProcess,     0,  MPI_COMM_WORLD );
      MPI_Send( next[lastRow-1],  ROWSIZE, MPI_CHAR,    nextProcess,     0,  MPI_COMM_WORLD );
      MPI_Recv( next[lastRow-1],  ROWSIZE, MPI_CHAR,    prevProcess,     0,  MPI_COMM_WORLD, &status );
      MPI_Recv( next[firstRow],   ROWSIZE, MPI_CHAR,    nextProcess,     0,  MPI_COMM_WORLD, &status );
    } else { //odd
      MPI_Recv( next[firstRow],   ROWSIZE, MPI_CHAR,    nextProcess,     0,  MPI_COMM_WORLD, &status );
      MPI_Recv( next[lastRow-1],  ROWSIZE, MPI_CHAR,    prevProcess,     0,  MPI_COMM_WORLD, &status );
      MPI_Send( next[lastRow-1],  ROWSIZE, MPI_CHAR,    nextProcess,     0,  MPI_COMM_WORLD );
      MPI_Send( next[firstRow],   ROWSIZE, MPI_CHAR,    prevProcess,     0,  MPI_COMM_WORLD );
    }

    // copy next to dish
    temp = dish;
    dish = next;
    next = temp;
  }
  
 
  
  if(rank == 0 && i == gens){
    gettimeofday(&tv,NULL);
    stop = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
    timespend = stop - start;
  }

  print(dish, firstRow, lastRow);
  MPI_Barrier(MPI_COMM_WORLD);
  printf( "Process %d done.  Exiting\n", rank );

  MPI_Finalize();
  if(rank == 0 && i == gens){
    printf("Program terminated....\n");
    printf("using %.8g milliseconds\n", timespend);   
    }

  return 0;
}
