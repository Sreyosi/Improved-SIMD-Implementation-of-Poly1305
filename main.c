/*
 ****************************************************************************
 Name        : modified_main.c						    *	
 ****************************************************************************
 Contributors of Improved SIMD Implementation of Poly1305 : 		    *
 Sreyosi Bhattacharyya,		Indian Statistical Institute 		    *
 Palash Sarkar        ,		Indian Statistical Institute                *
 ****************************************************************************
 Copyright (c) 2019, Sreyosi Bhattacharyya, Palash Sarkar                   *
 Permission to use this code is granted, provided the sources is mentioned. *
 Using, modifying, extracting from this code and/or algorithm(s)            *
 requires appropriate referencing.  					    *
 ****************************************************************************
 DISCLAIMER:								    *
 THIS SOFTWARE IS PROVIDED BY THE CONTRIBUORS AND THE COPYRIGHT OWNERS      *
 ``AS IS''. ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED *
 TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR *
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS OR THE COPYRIGHT*
 OWNERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, *
 OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF    *
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS   * 
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN    *
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)    *
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE * 
 POSSIBILITY OF SUCH DAMAGE.                                                *
 ****************************************************************************
 Description : Main function for Improved SIMD Implementation of Poly1305   *
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include<sys/time.h>
#include "poly1305/poly1305.h"
#include "measure/modified_perf.h"
#define MAXCTLEN	16*1024

void get_result1(size_t N,const unsigned char *key,const unsigned char*ct){

	/*some variables required to compute the tag, check the correctness of the new algorithm and recording the cycles/byte measurement*/	
	unsigned char tag[16];//,test[16];
	//int ret=0;
	double perf_poly1305_avx2_modified1=0;
        FILE*fp;
	poly1305_state_avx2 state_avx2;
        //poly1305_state state;
    
        /************************************************* COMPUTING TAG AND MEASURING CYCLES/BYTE ***********************************************************************************/
       MEASURE(
        CRYPTO_poly1305_init_avx2(&state_avx2, key);
        if(N%64==0 && N!=0){
		
        	CRYPTO_poly1305_update_avx2(&state_avx2, &ct[0],N, tag);
                
        }
        else if(N<49){
                CRYPTO_poly1305_finish_avx2(&state_avx2, &ct[0],N,tag);
	}
	else{
        	CRYPTO_poly1305_modified_update_avx2(&state_avx2, &ct[0],N,tag);
		
        }
        );
        
      
        /********************************** recording cycles/byte measurement ******************************************************************************************************/
	GET_CYCLES(perf_poly1305_avx2_modified1);
   
        fp=fopen("cycles_per_byte.txt","a");
    
        fprintf(fp,"%lu,%.2f\n", (unsigned long)N,(double)perf_poly1305_avx2_modified1/N);
    	fclose(fp);
        printf("%lu bytes,%.2f\n", (unsigned long)N,(double)(perf_poly1305_avx2_modified1)/N);
  	/*****************************************************************************************************************************************************************************/
	return ;
	
}


int main(int argc, char *argv[]) {
                unsigned char key[32];
                size_t i;
		size_t length_of_message_in_bytes;
         
   		length_of_message_in_bytes=(size_t)atoi(argv[1]);
                 
                srand(time(0));

		// key
		for(i=0; i<32; i++) {
			key[i] = rand() % 0xff;
		}


                // input message/ciphertext

		unsigned char *ct;
	        
		ct=(unsigned char*)calloc(length_of_message_in_bytes,sizeof(unsigned char));

        	for(i=0; i<length_of_message_in_bytes; i++) {
			ct[i] = rand() % 0xff;
		}

		get_result1(length_of_message_in_bytes,key,ct);
	        
                free(ct);                 
		return 0;
}






