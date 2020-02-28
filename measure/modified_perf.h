/*
 ============================================================================
 Name        : perf.h
 
 (C) M. Goll and S. Gueron
 Permission to use this code is granted, provided the sources is mentioned.
 Using, modifying, extracting from this code and/or algorithm(s)   
 requires appropriate referencing.  
 ***
 DISCLAIMER:
 THIS SOFTWARE IS PROVIDED BY THE CONTRIBUORS AND THE COPYRIGHT OWNERS 
 ``AS IS''. ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
 TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS OR THE COPYRIGHT
 OWNERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
 OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS   
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 POSSIBILITY OF SUCH DAMAGE.
 ***

 Description : This header file provides two macros for performance measuring in cycles.
 The measuring strategy is from Shay Gueron and Vlad Krasnov [1].
 The getticks() function is declared in cycle.h which is written by Matteo Frigo.

 [1] S. Gueron, V. Krasnov, Parallelizing message schedules to accelerate the computations of hash functions, Journal of Cryptographic Engineering 4:1-13 (2012)
 ============================================================================
 */

#ifndef PERF_H_
#define PERF_H_

#include <unistd.h>
#include "cycle.h"

// repeat the execution of the function(s) 25K times for warm-up
#define WARMUP_REPS 200000

// repeat the execution of the function(s) 100K times for measuring
#define MEASURE_REPS 400000

ticks START, END;
double RESULT,MEDIAN,time_record[MEASURE_REPS],temp,MIN_RESULT;
int W_CNT, M_CNT, R_CNT,smallest;

// do the measurement
int COMP(const void*a,const void *b){
	return ( *(double*)a - *(double*)b );
}


#define MEASURE(f)\
		MIN_RESULT = 1.7976931348623158e+308;\
		for(R_CNT=0;R_CNT < 5;R_CNT++){\
		for(W_CNT=0; W_CNT < WARMUP_REPS; W_CNT++) {f}	\
		for(M_CNT=0; M_CNT < MEASURE_REPS; M_CNT++) {	\
		START = getticks();								\
		f												\
		END = getticks();								\
		RESULT = (double)(END - START);					\
        time_record[M_CNT]=RESULT;						\
		}												\
        												\
		qsort(time_record,MEASURE_REPS,sizeof(double),COMP);\
		if(MEASURE_REPS%2)								\
			MEDIAN=time_record[MEASURE_REPS/2];			\
		else											\
			MEDIAN=(time_record[MEASURE_REPS/2]+time_record[(MEASURE_REPS/2)+1])/2;\
		if(MEDIAN<MIN_RESULT)\
			MIN_RESULT=MEDIAN;\
        }
																					
																					
// get the measurement result
#define GET_CYCLES(c)	c = MIN_RESULT;

#endif /* PERF_H_ */
