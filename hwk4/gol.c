#include <omp.h>
#include <stdio.h>
#include <sys/time.h>

#define GCOL 20
#define GROW 10

char *DISH0[GROW];
char *DISH1[GROW];
char *defau[GROW] = {
  "***OOOO*OO****OOO**O",
  "O**OOO**OOO****O****",
  "OO*OOO***********O**",
  "**O**O***O**********",
  "****OO**************",
  "*O*******OOO*OOO****",
  "****OO***O****OOO**O",
  "O******O**O******O**",
  "*O******OOO******O**",
  "****OOO**OO******O**",
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
  for( i = 0; i < GROW ; i++ ){
    DISH0[i] = (char *) malloc( (strlen( defau[0] ) + 1 ) * sizeof( char ) );
    strcpy( DISH0[i], defau[i] );

    DISH1[i] = (char *) malloc( (strlen( defau[0] ) + 1 ) * sizeof( char ) );
    strcpy( DISH1[i], defau[i] );
  }
}

void print(char ** dish){
    char ANSI_HOME[] = "\x1b[H";
    printf( "%s", ANSI_HOME );
    int i;

    for(i = 0 ; i < GROW ; i++ ){
       printf( "%s\n", dish[i] );
    }
    printf("Running......\n");
}
void life(char ** current, char ** next){
    int row, col;

    for(row = 0 ; row < GROW ; row++ ){
      for(col = 0 ; col < GCOL ; col++ ){
         int r, c, neighbor = 0;
         char temp = current[row][col];

         for(r = row - 1; r <= row + 1 ; r++ ){
            if(r == -1 || r == GROW)continue;
            for(c = col - 1; c <= col + 1; c++){
               if(c == -1|| c == GCOL)continue;
               if(c == col && r == row)continue;

               if(current[r][c] == 'O')neighbor++;
             }
         }
        //populated or underpopulated
         if(temp == 'O'){ // if this cell lives
            if(neighbor < 2 || neighbor > 3){
              next[row][col] = '*';
            }else{
              next[row][col] = 'O';
            }
         }

         if(temp == '*'){// if this cell died with 3 neighbors
            if(neighbor == 3){
              next[row][col] = 'O';
           }else{
            next[row][col] = '*';
           }
         }
      }
   }
}

int main(){
  char **dish, **next;
  int i, gens = 10;
  struct timeval tv;
  double start, stop, timespend;

  cleanScreen();
  initDish();

  dish = DISH0;
  next = DISH1;

  print(dish);
  gettimeofday(&tv,NULL);
  start = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;

  for(i = 0 ; i < gens ; i++){
     sleep(1);
     life(dish, next);
     print(next);

     dish = next;
  }

  gettimeofday(&tv,NULL);
  stop = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
  timespend = stop - start - gens*1000;
  printf("using %.8g milliseconds\n", timespend);

  printf("Program terminated....\n");
  return 0;
}
