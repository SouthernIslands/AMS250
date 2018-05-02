/*
* Parallel on inner loop
*/
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define N 1600 //Scale at 100 has already took me 3 seconds

int a[N][N];
int b[N][N];
int res[N][N];

int main(){
  int i, j, k, temp;
  int tid, nthreads;
  struct timeval tv;
  double start, stop, timespend;

  //preparation
  int c = 1,d = 2;
  for (i = 0; i<N ; i++){
      for (j = 0; j<N ; j++){
            a[i][j] = c++;
            b[i][j] = d++;
     	}
}
  gettimeofday(&tv,NULL);
  start = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;

  for(i = 0; i < N ;i++){
    for(j = 0; j < N; j++){
    #pragma omp parallel for private(k,tid) shared(a,b,res)
      for(k = 0; k < N; k++){
        tid = omp_get_thread_num();
        res[i][j]  += a[i][k]*b[k][j];
        if(tid == 0){
                nthreads = omp_get_num_threads();
//                printf("Number of threads = %d\n", nthreads);
                }
      }
    }
  }

  gettimeofday(&tv,NULL);
  stop = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
  timespend = stop - start;
  printf("Computing on %d threads and using %.8g milliseconds\n", nthreads, timespend);

}
