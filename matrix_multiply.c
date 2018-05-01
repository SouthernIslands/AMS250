#include <omp.h>
#include <stdio.h>
#include <time.h>

#define N 2

int a[N][N] = {
  {1,2,3},
  {1,2,3},
  {1,2,3}
};

int b[N][N] = {
  {3,2,1},
  {3,2,1},
  {3,2,1}
};

int res[N][N];

int main(){
  int i, j, k, temp;
  struct timeval tv;
  double start, stop, timespend;

  gettimeofday(&tv,NULL);
  start = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
  #pragma omp parallel for private(i,j,k) shared(a,b,res);
  for(i = 0; i < N ;i++){
    for(j = 0; j < N; j++){
      for(k = 0; k < N; k++){
        res[i][j]  += a[i][k]*b[k][j];
      }
    }
  }

  gettimeofday(&tv,NULL);
  stop = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
  timespend = stop - start;
  printf("using %.8g milliseconds\n", timespend);

}
