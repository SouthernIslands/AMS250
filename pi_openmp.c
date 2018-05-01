#include <omp.h>
#include <stdio.h>

static long num_steps = 100000000;

int main(){
double x, pi, step, timespend, sum=0.0;
int i;
step = 1./(double)num_steps;
gettimeofday(&tv,NULL);

start = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
#pragma omp parallel for private(x) reduction(+:sum)
  for(i = 0; i < num_steps; i++){
    x = (i + .5)*step;
    sum  = sum + 1.0/(1.+x*x);
  }
pi = 4.0*sum*step;
gettimeofday(&tv,NULL);
stop = (tv.tv_sec)*1000 + (tv.tv_usec)/1000;
timespend = stop - start;

printf("Compute PI is %.16g\n", pi);
printf("using %.4g seconds\n", timespend);

}
