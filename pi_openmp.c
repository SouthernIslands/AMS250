#include <omp.h>
#include <stdio.h>
#include <time.h>

static long num_steps = 100000000;

int main(){
double x, pi, step, timespend, sum=0.0;
double start, stop, timespend;
struct timeval tv;
int i,tid,nthreads;

step = 1./(double)num_steps;
gettimeofday(&tv,NULL);

start = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
#pragma omp parallel for private(x,tid) reduction(+:sum)
  for(i = 0; i < num_steps; i++){
    // tid = omp_get_thread_num();
    // if(tid == 0){
    //   nthreads = omp_get_num_threads();
    //   printf("Number of threads = %d\n", nthreads);
    // }
    // printf("Computing on thread %d\n", tid); may slow down the program?
    x = (i + .5)*step;
    sum  = sum + 1.0/(1.+x*x);
  }
printf("Computing on thread %d\n", nthreads);
pi = 4.0*sum*step;
gettimeofday(&tv,NULL);
stop = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
timespend = stop - start;

printf("Compute PI is %.16g\n", pi);
printf("using %.8g milliseconds\n", timespend);

}
