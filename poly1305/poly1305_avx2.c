/*
 ****************************************************************************
 Name        : poly1305_avx2.c						    *	
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
									    *
 Description: This code claims to be an improved SIMD implementation of     *
	      Poly1305. This improvement is with respect to the best known  *
              SIMD implementation of Poly1305 which is due to Shay Gueron   *
              and Martin Goll. This improved Poly1305 implementation is     *
              intended to run on any x86 or x86_x64 platform with AVX2      *
              enabled. There are no alignment requirements for the input    *
              data, key, mac and state buffer pointers. The internal state  *
              buffer needs a size of at least 519 bytes. All the functions  *
              which have been claimed to be modified originate from SIMD    *
              code of Poly1305 developed by Shay Gueron and Martin Goll.    *
									    *
	      The functions align4x128_init, vec256x5 mul4x130R_rem1, *
              mul4x130R_rem2, load4x128_test, load4x128_test1 and           *
              align4x128_large have been introduced in this modified        * 
              version. The function CRYPTO_poly1305_modified_update_avx2    *                            
	      does the updation of state in each iteration just as 	    * 
              CRYPTO_poly1305_update_avx2 but it works according to the need* 
              of the modified algorithm. All these changes lead to 	    *
              CRYPTO_poly1305_modified_finish_avx2 function which is highly *
              simplified version of CRYPTO_poly1305_finish_avx2.	    *	     
 ****************************************************************************
 */
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>
#include "poly1305.h"

#if defined(__AVX2__)

typedef unsigned vec128 __attribute__ ((vector_size (16)));
typedef unsigned vec256 __attribute__ ((vector_size (32)));
typedef struct {
	vec256 v0;
	vec256 v1;
} vec256x2;
typedef struct {
	vec256 v0;
	vec256 v1;
	vec256 v2;
} vec256x3;
typedef struct {
	vec256 v0;
	vec256 v1;
	vec256 v2;
	vec256 v3;
	vec256 v4;
} vec256x5;
typedef struct {
	vec256 k;              /*  32 bytes  */
	vec256 r1;             /*  32 bytes  */
	vec256 r2;             /*  32 bytes  */
	vec256 r4;             /*  32 bytes  */
	vec256 r15;            /*  32 bytes  */
	vec256 r25;            /*  32 bytes  */
	vec256 r45;            /*  32 bytes  */
	vec256x2 m;            /*  64 bytes  */
	vec256x3 p;            /*  96 bytes  */
	union {                /*  64 bytes  */
		uint8_t b[64];
		vec256x2 v;
		vec128 va[4];
	}buffer;
	uint32_t p_init;       /*   4 bytes  */
	uint32_t leftover;     /*   4 bytes  */
}poly1305_state_internal_avx2; /* 456 bytes total + 63 bytes for alignment = 519 bytes raw */

#define LOAD128(m)                     (vec128)_mm_loadu_si128((__m128i*)(m))
#define LOAD256(m)                     (vec256)_mm256_loadu_si256((__m256i*)(m))
#define LOAD256_COMBINE(x,y)	       (vec256)_mm256_loadu2_m128i((__m128i*)(x),(__m128i*)(y))
#define STORE128(m,r)                  _mm_storeu_si128((__m128i*)(m), (__m128i) (r))
#define STORE256(m,r)                  _mm256_storeu_si256((__m256i*)(m), (__m256i) (r))
#define TC256(x)                       (vec256)_mm256_castsi128_si256((__m128i) (x))
#define TC128(x)                       (vec128)_mm256_castsi256_si128((__m256i) (x))
#define BC32R(x)                       (vec256)_mm256_broadcastd_epi32((__m128i) (x))
#define BC32M(x)                       (vec256){x,x,x,x,x,x,x,x}
#define BC32Z(x)                       (vec256){x,0,x,0,x,0,x,0}
#define BC128P(x)                      (vec256)_mm256_permute2x128_si256((__m256i) (x), (__m256i) (x), 0)
#define BC128P1(x)                     (vec256)_mm256_permute2x128_si256((__m256i) (x), (__m256i) (x), 1)
#define BC128P2(x,y)                   (vec256)_mm256_permute2x128_si256((__m256i) (x), (__m256i) (y), 3)
#define BC32M1(x)                      (vec256){0,x,0,x,0,x,x,x}
#define BC32M2(x)                      (vec256){0,x,0,x,x,x,x,x}
#define BC32M3(x)                      (vec256){0,x,x,x,x,x,x,x}
#define BCZERO(x)		       (vec256){0,0,0,0,0,0,0,x}
#define EXT128(x,pos)                  (vec128)_mm256_extractf128_si256((__m256i) (x), pos)
#define EXT64(x,pos)                           _mm256_extract_epi64((__m256i)x,pos)
#define INS128(y,x,pos)                (vec256)_mm256_inserti128_si256((__m256i) (y), (__m128i) (x), (pos))
#define SET02(x3,x2,x1,x0)             (((x3) << 6) | ((x2) << 4) | ((x1) << 2) | (x0))
#define SET32(x7,x6,x5,x4,x3,x2,x1,x0) (vec256)_mm256_set_epi32(x7,x6,x5,x4,x3,x2,x1,x0)
#define SET64(x3,x2,x1,x0)             (vec256)_mm256_set_epi64x(x3,x2,x1,x0)
#define UNPACK_LOW32(a,b)              (vec256)_mm256_unpacklo_epi32((__m256i) (a), (__m256i) (b))
#define UNPACK_HIGH32(a,b)             (vec256)_mm256_unpackhi_epi32((__m256i) (a), (__m256i) (b))
#define UNPACK_LOW64(a,b)              (vec256)_mm256_unpacklo_epi64((__m256i) (a), (__m256i) (b))
#define UNPACK_HIGH64(a,b)             (vec256)_mm256_unpackhi_epi64((__m256i) (a), (__m256i) (b))
#define MUL32(x,y)                     (vec256)_mm256_mul_epu32((__m256i) (x), (__m256i) (y))
#define ADD32(x,y)                     (vec256)_mm256_add_epi32((__m256i) (x), (__m256i) (y))
#define ADD64(x,y)                     (vec256)_mm256_add_epi64((__m256i) (x), (__m256i) (y))
#define MPLX(x,y,msk)                  (vec256)_mm256_blend_epi32((__m256i)x, (__m256i) (y), msk)
#define PERM32(x,ord)                  (vec256)_mm256_permutevar8x32_epi32((__m256i) (x), (__m256i) (ord))
#define PERM64(x,ord)                  (vec256)_mm256_permute4x64_epi64((__m256i) (x), ord)
#define SRI32(x,cnt)                   (vec256)_mm256_srli_epi32((__m256i) (x), (cnt))
#define SRV32(x,cnt)                   (vec256)_mm256_srlv_epi32((__m256i) (x), (__m256i) (cnt))
#define SLI32(x,cnt)                   (vec256)_mm256_slli_epi32((__m256i) (x), (cnt))
#define SLV32(x,cnt)                   (vec256)_mm256_sllv_epi32((__m256i) (x), (__m256i) (cnt))
#define SRI64(x,cnt)                   (vec256)_mm256_srli_epi64((__m256i) (x), (cnt))
#define SRV64(x,cnt)                   (vec256)_mm256_srlv_epi64((__m256i) (x), (__m256i) (cnt))
#define SLI64(x,cnt)                   (vec256)_mm256_slli_epi64((__m256i) (x), (cnt))
#define OR256(x,y)                     (vec256)_mm256_or_si256((__m256i) (x), (__m256i) (y))
#define AND256(x,y)                    (vec256)_mm256_and_si256((__m256i) (x), (__m256i) (y))
#define ALIGN26(x)                                                                        \
            AND256(                                                                       \
               OR256(                                                                     \
                 SLV32(x, SET32(32,32,32,24,18,12,6,0)),                                  \
                 PERM32(SRV32(x, SET32(32,32,32,2,8,14,20,26)), SET32(6,5,4,3,2,1,0,7))), \
               SET32(0,0,0,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff))
#define SHUFFLE(x,index)               (vec256)_mm256_shuffle_epi32 ((__m256i) (x), index)
#define SHIFT128_LEFT(x, cnt)	       (vec256)_mm256_slli_si256 ((__m256i) (x), (cnt))
#define SHIFT128_RIGHT(x, cnt)         (vec256)_mm256_srli_si256 ((__m256i) (x), (cnt))
#define MERGE(x, y)		       (vec256)_mm256_setr_m128i ((__m128i) (x), (__m128i) (y))
#define MASK(x)			       (vec256){0xffffffff,0xffffffff,0xffffffff,0xffffffff,0,0,0,0}
#define BC32M_L(x)		       (vec256){x,x,x,x,x,x,0,x}

/* functions for processing 4x 130 bits */


static inline
vec256x2 load4x128_test(unsigned int *ip,int rem) {
	   if(rem==1)
     		return (vec256x2){SET32(0,0,0,0,0,0,0,0), BC128P2(LOAD256(&ip[0]),SET32(0,0,0,0,0,0,0,0))};
          if(rem==2)
    	        return (vec256x2){SET32(0,0,0,0,0,0,0,0), LOAD256(&ip[0])};
	  if(rem==3)
		return (vec256x2){BC128P2(LOAD256(&ip[0]),SET32(0,0,0,0,0,0,0,0)), LOAD256(&ip[4])};
	  else
		return (vec256x2){LOAD256(&ip[0]), LOAD256(&ip[8])};
		
}


vec256x2 load4x128_test1(const unsigned int *ip, size_t bytes) {
          uint8_t* src=(uint8_t*)ip;
          vec256x2 ret;
                
         ret.v0=AND256(TC256(LOAD128(&ip[8])),MASK(0xffffffff));
         ret.v1=PERM64(AND256(TC256(LOAD128(&src[48-bytes])),MASK(0xffffffff)),0x4F);
         
         ret.v1=OR256(ret.v0,ret.v1);
	 ret.v0=LOAD256(&ip[0]);
         return ret;  
    
}


static inline
vec256x2 load4x128(unsigned int *ip) {
	return (vec256x2){LOAD256(&ip[0]), LOAD256(&ip[8])};
      
}


static inline
vec256x3 align4x128_last(vec256x2 x,size_t bytes) {
	vec256 msk = BC32M(0x3ffffff);
	vec256 pad = BC32M_L(1<<24);
	vec256x3 ret;

	ret.v1=OR256(SHIFT128_RIGHT(x.v1,1),BCZERO(0x1000000));
        bytes--;        

        if (bytes &  8) { ret.v1=SHIFT128_RIGHT(ret.v1, 8);}
	if (bytes &  4) { ret.v1=SHIFT128_RIGHT(ret.v1, 4);}
	if (bytes &  2) { ret.v1=SHIFT128_RIGHT(ret.v1, 2);}
	if (bytes &  1) { ret.v1=SHIFT128_RIGHT(ret.v1, 1);}
  
        x.v1=MPLX(x.v1,ret.v1,0xf0);
       
	ret.v0 = PERM64(UNPACK_HIGH64(x.v0, x.v1), SET02(3,1,2,0));
	x.v0 = PERM64(UNPACK_LOW64(x.v0, x.v1), SET02(3,1,2,0));
	ret.v2 = OR256(SRI64(ret.v0, 40),  pad);
	x.v1 = OR256(SRI64(x.v0, 46), SLI64(ret.v0, 18));
	ret.v1 = AND256(MPLX(SRI64(x.v0, 26), x.v1, 0xAA), msk);
	ret.v0 = AND256(MPLX(x.v0, SLI64(x.v1, 26), 0xAA), msk);
    	//ret.v2[6]=ret.v2[6]&(0x0ffffff);
    	return ret;  
}


static inline
vec256x3 align4x128_init(vec256x2 x,int rem) {
    //vec256 msk = BC32M(0x3ffffff);
    vec256 pad;
    vec256x3 ret;
    uint32_t i;
   
   if(rem==2||rem==1){
    //vec256x3 ret;
    i=1<<24;
    vec256 msk = SET32(0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff);
    //vec256 pad=SET32(i,i,i,i,i,i,i,i);
    pad=SET32(i,i,i,i,i,i,i,i);
    x.v0=MPLX(SLI64(x.v1,6),x.v1,0xDD);   
    x.v0=MPLX(SLI64(x.v1,18),x.v0,0x77); 
    x.v1=SHUFFLE(x.v1,0xC9);
    x.v0=MPLX(SHUFFLE(SLI64(x.v1,12),0xD2),x.v0,0xBB);
    ret.v0=PERM32(AND256(x.v0,msk),SET32(3,7,2,6,1,5,0,4));
    if(rem==2)
    	ret.v1=PERM64(AND256(OR256(SRI64(x.v1, 40),  pad),SET32(0,0xffffffff,0,0,0,0xffffffff,0,0)),SET02(3,1,2,0));
    else
	ret.v1=PERM64(AND256(OR256(SRI64(x.v1, 40),  pad),SET32(0,0xffffffff,0,0,0,0,0,0)),SET02(3,1,2,0));
    return ret;
   }

   //else{
   	    if(rem==3)
	 		pad = BC32M3(1<<24);
   	    else
    		pad= BC32M(1<<24);
        ret.v0 = PERM64(UNPACK_HIGH64(x.v0, x.v1), SET02(3,1,2,0));
	x.v0 = PERM64(UNPACK_LOW64(x.v0, x.v1), SET02(3,1,2,0));
  //  }
        ret.v2 = OR256(SRI64(ret.v0, 40),  pad);
	x.v1 = OR256(SRI64(x.v0, 46), SLI64(ret.v0, 18));
  	
        vec256 msk=(vec256){0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff};
        ret.v1 = AND256(MPLX(SRI64(x.v0, 26), x.v1, 0xAA), msk);
        ret.v0 = AND256(MPLX(x.v0, SLI64(x.v1, 26), 0xAA), msk);
    return ret;     
}

/*vec256x3 align4x128_small(vec256x2 x,int rem) {
	vec256 msk = BC32M(0x3ffffff);
    vec256 pad;
    pad = BC32M(1<<24);
	vec256x3 ret;
	ret.v0 = PERM64(UNPACK_HIGH64(x.v0, x.v1), SET02(3,1,2,0));
	x.v0 = PERM64(UNPACK_LOW64(x.v0, x.v1), SET02(3,1,2,0));
	ret.v2 = OR256(SRI64(ret.v0, 40),  pad);
	x.v1 = OR256(SRI64(x.v0, 46), SLI64(ret.v0, 18));
	ret.v1 = AND256(MPLX(SRI64(x.v0, 26), x.v1, 0xAA), msk);
	ret.v0 = AND256(MPLX(x.v0, SLI64(x.v1, 26), 0xAA), msk);
	ret.v2[6]=ret.v2[6]&(0x0ffffff);
   
    return ret;
   
}*/

/*vec256x3 align4x128_large(vec256x2 x,int rem) {
	vec256 msk = BC32M(0x3ffffff);
    	vec256 pad;
    	pad = BC32M(1<<24);
	vec256x3 ret;
	ret.v0 = PERM64(UNPACK_HIGH64(x.v0, x.v1), SET02(3,1,2,0));
	x.v0 = PERM64(UNPACK_LOW64(x.v0, x.v1), SET02(3,1,2,0));
	ret.v2 = OR256(SRI64(ret.v0, 40),  pad);
	if(rem==1){
  		ret.v2=AND256(OR256(SRI64(ret.v0, 40),  pad),SET32(0,0xffffffff,0,0,0,0,0,0));
       		//return ret;
        }
        else if(rem==2){
   			ret.v2=AND256(OR256(SRI64(ret.v0, 40),  pad),SET32(0,0xffffffff,0,0xffffffff,0,0,0,0));
       			//return ret;
        }
        else if(rem==3){
			ret.v2=AND256(OR256(SRI64(ret.v0, 40),  pad),SET32(0,0xffffffff,0,0xffffffff,0,0xffffffff,0,0));
                        //return ret;
        }
	x.v1 = OR256(SRI64(x.v0, 46), SLI64(ret.v0, 18));
	ret.v1 = AND256(MPLX(SRI64(x.v0, 26), x.v1, 0xAA), msk);
	ret.v0 = AND256(MPLX(x.v0, SLI64(x.v1, 26), 0xAA), msk);

	//else 
        return ret;
		
}*/
vec256x3 align4x128_large(vec256x2 x,int rem) {
	vec256 msk = BC32M(0x3ffffff);
    	vec256 pad;
    	pad = BC32M(1<<24);
	vec256x3 ret;
	ret.v0 = PERM64(UNPACK_HIGH64(x.v0, x.v1), SET02(3,1,2,0));
	x.v0 = PERM64(UNPACK_LOW64(x.v0, x.v1), SET02(3,1,2,0));
	ret.v2 = OR256(SRI64(ret.v0, 40),  pad);
	x.v1 = OR256(SRI64(x.v0, 46), SLI64(ret.v0, 18));
	ret.v1 = AND256(MPLX(SRI64(x.v0, 26), x.v1, 0xAA), msk);
	ret.v0 = AND256(MPLX(x.v0, SLI64(x.v1, 26), 0xAA), msk);
        if(rem==1){
  		ret.v2=AND256(ret.v2,SET32(0,0xffffffff,0,0,0,0,0,0));
       		return ret;
        }
        else if(rem==2){
   			ret.v2=AND256(ret.v2,SET32(0,0xffffffff,0,0xffffffff,0,0,0,0));
       			return ret;
        }
        else if(rem==3){
			ret.v2=AND256(ret.v2,SET32(0,0xffffffff,0,0xffffffff,0,0xffffffff,0,0));
                        return ret;
       }
	else return ret;
	
}


static inline
vec256x3 align4x128(vec256x2 x) {
	vec256 msk = BC32M(0x3ffffff);
	vec256 pad = BC32M(1<<24);
	vec256x3 ret;
	ret.v0 = PERM64(UNPACK_HIGH64(x.v0, x.v1), SET02(3,1,2,0));
	x.v0 = PERM64(UNPACK_LOW64(x.v0, x.v1), SET02(3,1,2,0));
	ret.v2 = OR256(SRI64(ret.v0, 40),  pad);
	x.v1 = OR256(SRI64(x.v0, 46), SLI64(ret.v0, 18));
	ret.v1 = AND256(MPLX(SRI64(x.v0, 26), x.v1, 0xAA), msk);
	ret.v0 = AND256(MPLX(x.v0, SLI64(x.v1, 26), 0xAA), msk);
	return ret;
}
static inline
vec256x2 hadd4x130(vec256x5 x) {
	vec256x2 ret;
	ret.v0 = ADD64(UNPACK_HIGH64(x.v0, x.v1), UNPACK_LOW64(x.v0, x.v1));
	ret.v1 = ADD64(UNPACK_HIGH64(x.v2, x.v3), UNPACK_LOW64(x.v2, x.v3));
	ret.v0 = ADD64(INS128(ret.v0, TC128(ret.v1), 1), INS128(ret.v1, EXT128(ret.v0, 1), 0));
	ret.v1 = ADD64(x.v4, PERM64(x.v4, SET02(1,0,3,2)));
	ret.v1 = ADD64(ret.v1, PERM64(ret.v1, SET02(0,0,0,1)));
	return ret;
}
static inline
vec256x3 add4x130(vec256x3 x, vec256x3 y) {
	return (vec256x3){ADD32(x.v0, y.v0),ADD32(x.v1, y.v1),ADD32(x.v2, y.v2)};
}
static inline
vec256x3 red4x130(vec256x5 x) {
	vec256 msk = BC32Z(0x3ffffff);
	x.v1 = ADD64(x.v1, SRI64(x.v0, 26)); x.v0 = AND256(x.v0, msk);
	x.v4 = ADD64(x.v4, SRI64(x.v3, 26)); x.v3 = AND256(x.v3, msk);
	x.v2 = ADD64(x.v2, SRI64(x.v1, 26)); x.v1 = AND256(x.v1, msk);
	x.v0 = ADD64(x.v0, MUL32(SRI64(x.v4, 26), BC32Z(5)));
	x.v4 = AND256(x.v4, msk);
	x.v3 = ADD64(x.v3, SRI64(x.v2, 26)); x.v2 = AND256(x.v2, msk);
	x.v1 = ADD64(x.v1, SRI64(x.v0, 26)); x.v0 = AND256(x.v0, msk);
	x.v4 = ADD64(x.v4, SRI64(x.v3, 26)); x.v3 = AND256(x.v3, msk);
	x.v0 = MPLX(x.v0, SLI64(x.v2, 32), 0xAA);
	x.v1 = MPLX(x.v1, SLI64(x.v3, 32), 0xAA);
	return (vec256x3){x.v0,x.v1,x.v4};
}
static inline
vec256x5 mul4x130M(vec256x3 x, vec256x2 m, vec256 r1) {
	vec256x5 ret;
	// r2[0],r2[1],r4[4],r3[4],r4[1],r3[1],r4[0],r3[0]
	ret.v0 = UNPACK_LOW32(m.v0, m.v1);
	// r4[4],r2[4],r2[2],r2[3],r4[3],r3[3],r4[2],r3[2]
	ret.v1 = UNPACK_HIGH32(m.v0, m.v1);

	vec256x3 t;
	// r1[1],r1[0],r2[1],r2[0],r3[1],r3[0],r4[1],r4[0]
	vec256 ord = SET32(1,0,6,7,2,0,3,1);
	t.v0 = MPLX(PERM32(r1, ord), PERM32(ret.v0, ord), 0x3F);
	// r1[3],r1[2],r2[3],r2[2],r3[3],r3[2],r4[3],r4[2]
	ord = SET32(3,2,4,5,2,0,3,1);
	t.v1 = MPLX(PERM32(r1, ord), PERM32(ret.v1, ord), 0x3F);
	// r1[1],r1[4],r2[1],r2[4],r3[1],r3[4],r4[1],r4[4]
	ord = SET32(1,4,6,6,2,4,3,5);
	t.v2 = MPLX(MPLX(
			PERM32(r1, ord),
			PERM32(ret.v1, ord), 0x10),
			PERM32(ret.v0, ord), 0x2F);

	ret.v0 = MUL32(x.v0, t.v0);
	ret.v1 = MUL32(x.v1, t.v0);
	ret.v2 = MUL32(x.v0, t.v1);
	ret.v3 = MUL32(x.v1, t.v1);
	ret.v4 = MUL32(x.v0, t.v2);
	ord = SET32(6,7,4,5,2,3,0,1);
	m.v0 = PERM32(t.v0, ord);
	m.v1 = PERM32(t.v1, ord);
	ret.v4 = ADD64(ret.v4, MUL32(x.v2, t.v0));
	ret.v4 = ADD64(ret.v4, MUL32(x.v1, m.v1));
	ret.v1 = ADD64(ret.v1, MUL32(x.v0, m.v0));
	ret.v3 = ADD64(ret.v3, MUL32(x.v0, m.v1));
	ret.v2 = ADD64(ret.v2, MUL32(x.v1, m.v0));
	x.v0 = PERM32(x.v0, ord);
	ret.v2 = ADD64(ret.v2, MUL32(x.v0, t.v0));
	ret.v3 = ADD64(ret.v3, MUL32(x.v0, m.v0));
	ret.v4 = ADD64(ret.v4, MUL32(x.v0, t.v1));
	t.v1 = ADD32(t.v2, SLI32(t.v2, 2));
	m.v1 = ADD32(m.v1, SLI32(m.v1, 2));
	ret.v0 = ADD64(ret.v0, MUL32(x.v0, m.v1));
	ret.v0 = ADD64(ret.v0, MUL32(x.v1, t.v1));
	ret.v1 = ADD64(ret.v1, MUL32(x.v0, t.v1));
	ret.v2 = ADD64(ret.v2, MUL32(x.v2, m.v1));
	ret.v3 = ADD64(ret.v3, MUL32(x.v2, t.v1));
	x.v1 = PERM32(x.v1, ord);
	ret.v1 = ADD64(ret.v1, MUL32(x.v1, m.v1));
	ret.v2 = ADD64(ret.v2, MUL32(x.v1, t.v1));
	ret.v3 = ADD64(ret.v3, MUL32(x.v1, t.v0));
	ret.v4 = ADD64(ret.v4, MUL32(x.v1, m.v0));
	m.v1 = PERM32(m.v1, ord);
	t.v1 = PERM32(t.v1, ord);
	ret.v0 = ADD64(ret.v0, MUL32(x.v1, m.v1));
	ret.v0 = ADD64(ret.v0, MUL32(x.v2, t.v1));
	ret.v1 = ADD64(ret.v1, MUL32(x.v2, m.v1));

	return ret;
}
static inline vec256x5 mul4x130R_rem1(vec256x3 x, vec256 y, vec256 z){
	vec256x5 ret;
      
    /*Multiplication 1*/
   
    ret.v0=MUL32(x.v0,PERM32(y,SET64(0,0,0,0)));
    
    /*Multiplication 2*/

    ret.v0=ADD64(ret.v0,MUL32(PERM64(x.v0,SET02(2,1,0,3)),PERM32(y,SET64(1,1,1,5))));

    /*Multiplication 3*/
   
   
    ret.v0=ADD64(ret.v0,MUL32(PERM64(x.v0,SET02(1,0,3,2)),PERM32(y,SET64(2,2,6,6))));

    /*Multiplication 4*/

    ret.v0=ADD64(ret.v0,MUL32(PERM64(x.v0,SET02(0,3,2,1)),PERM32(y,SET64(3,7,7,7))));

    /*Multiplication 5*/

    ret.v0=ADD64(ret.v0,MUL32(PERM64(x.v1,SET02(3,3,3,3)),MPLX(PERM32(y,SET64(7,6,5,0)),z,0x3)));

    
    
    /*Multiplication 6*/

    ret.v4=MUL32(x.v0,PERM32(y,SET64(1,2,3,4)));

    /*Multiplication 7*/

    vec256 temp=SET64(0,0,0,0);
    x.v0=PERM64(ADD64(UNPACK_LOW64(ret.v4,temp),UNPACK_HIGH64(ret.v4,temp)),SET02(2,0,1,1));
    ret.v4=ADD64(ADD64(UNPACK_LOW64(temp,x.v0),UNPACK_HIGH64(temp,x.v0)),MUL32(x.v1,PERM32(y,temp)));

    /*Formatting Output*/
    ret.v3=MPLX(temp,ret.v0,0xC0);
    ret.v0=AND256(ret.v0,SET32(0,0,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff));
    ret.v2=PERM64(ret.v0,SET02(2,3,3,3));
    ret.v1=PERM64(ret.v0,SET02(1,3,3,3));
    ret.v0=PERM64(ret.v0,SET02(0,3,3,3));
    
	return ret;   
}
static inline vec256x5 mul4x130R_rem2(vec256x3 x, vec256 y, vec256 z){
	vec256x5 ret;
        vec256 t1;

       //multiplication 1

	vec256 t0=PERM32(y,SET64(0,0,0,0));
	ret.v0=MUL32(x.v0,t0);//1st multiplicand
	
        x.v0=SHUFFLE(x.v0,0xB1);
	ret.v1=MUL32(x.v0,t0); //2nd multiplicand

	//multiplication 2

	t0=PERM32(y,SET32(1,1,1,1,1,1,5,5)); 
        x.v0=PERM64(x.v0,SET02(2,1,0,3));
	ret.v1=ADD64(ret.v1,MUL32(x.v0,t0));//1st multiplicand

	ret.v0=ADD64(ret.v0,MUL32(t0,SHUFFLE(x.v0,0XB1)));//2nd multiplicand

	//multiplilcation 3
	
	x.v0=PERM64(x.v0,SET02(2,1,0,3));
	t0=PERM32(y,SET32(2,2,2,2,6,6,6,6));
	ret.v1=ADD64(ret.v1,MUL32(x.v0,t0));//1st multiplicand

	ret.v0=ADD64(ret.v0,MUL32(SHUFFLE(x.v0,0xB1),t0));//2nd multiplicand
	
	
	//multiplication 4

	x.v0=PERM64(x.v0,SET02(2,1,0,3));
	t0=PERM32(y,SET32(3,3,7,7,7,7,7,7));
	ret.v1=ADD64(ret.v1,MUL32(x.v0,t0));//1st multiplicand
	
	ret.v0=ADD64(ret.v0,MUL32(t0,SHUFFLE(x.v0,0xB1)));//2nd multiplicand

	//multiplication 5

	t1=PERM32(x.v1,SET32(4,6,4,6,4,6,4,6)); 
        t0=MPLX(PERM32(y,SET32(7,7,6,6,5,5,0,0)),z,0x3);
        ret.v0=ADD64(ret.v0,MUL32(t1,t0));//1st multiplicand
        
	ret.v1=ADD64(ret.v1,MUL32(SHUFFLE(t1,0xB1),t0));//2nd multiplicand


	//multiplication 6

       t0=PERM32(y,SET32(4,4,1,1,2,2,3,3));
       t1=MUL32(x.v0,t0);//1st multiplicand
	 
       ret.v4=MUL32(SHUFFLE(x.v0,0xB1),t0);//2nd multiplicand

       /* Formatting ret.v4 */
    //-----------------------------------------------------------------------
	ret.v4=ADD64(UNPACK_HIGH64(ret.v4,t1),UNPACK_LOW64(ret.v4,t1));
	t0=SET64(0,0,0,0);
        ret.v4=PERM64(ret.v4,SET02(3,1,2,0));
        ret.v4=ADD64(UNPACK_HIGH64(ret.v4,t0),UNPACK_LOW64(ret.v4,t0));
        ret.v4=PERM64(ret.v4,SET02(0,2,1,1));
	
	//multiplication 7
	
	ret.v4=ADD64(ret.v4,MUL32(x.v1,PERM32(y,t0)));//both multiplicands

	/* Formatting the output */
	//-------------------------------------------------------------------------

	ret.v3=UNPACK_HIGH64(ret.v1,ret.v0);
        ret.v2=UNPACK_LOW64(ret.v1,ret.v0);
	ret.v1=MPLX(t0,ret.v3,0x0F);
	ret.v0=MPLX(t0,ret.v2,0x0F);	
	ret.v3=MPLX(t0,ret.v3,0xF0);
	ret.v2=MPLX(t0,ret.v2,0xF0);
	ret.v1=PERM64(ret.v1,SET02(1,0,3,2));
	ret.v0=PERM64(ret.v0,SET02(1,0,3,2));

        return ret;	
}

static inline
vec256x5 mul4x130R(vec256x3 x, vec256 y, vec256 z) {
	vec256x5 ret;
	vec256 ord = SET32(6,7,4,5,2,3,0,1);
	vec256 t0 = PERM64(y, SET02(0,0,0,0));
	vec256 t1 = PERM64(y, SET02(1,1,1,1));
	ret.v0 = MUL32(x.v0, t0);
	ret.v1 = MUL32(x.v1, t0);
	ret.v4 = MUL32(x.v2, t0);
	ret.v2 = MUL32(x.v0, t1);
	ret.v3 = MUL32(x.v1, t1);
	t0 = PERM32(t0, ord);
	t1 = PERM32(t1, ord);
	ret.v1 = ADD64(ret.v1, MUL32(x.v0, t0));
	ret.v2 = ADD64(ret.v2, MUL32(x.v1, t0));
	ret.v3 = ADD64(ret.v3, MUL32(x.v0, t1));
	ret.v4 = ADD64(ret.v4, MUL32(x.v1, t1));
	vec256 t2 = PERM64(y, SET02(2,2,2,2));
	ret.v4 = ADD64(ret.v4, MUL32(x.v0, t2));
	x.v0 = PERM32(x.v0, ord);
	x.v1 = PERM32(x.v1, ord);
	t2 = PERM32(t2, ord);
	ret.v0 = ADD64(ret.v0, MUL32(x.v1, t2));
	ret.v1 = ADD64(ret.v1, MUL32(x.v2, t2));
	ret.v3 = ADD64(ret.v3, MUL32(x.v0, t0));
	ret.v4 = ADD64(ret.v4, MUL32(x.v1, t0));
	t0 = PERM32(t0, ord);
	t1 = PERM32(t1, ord);
	ret.v2 = ADD64(ret.v2, MUL32(x.v0, t0));
	ret.v3 = ADD64(ret.v3, MUL32(x.v1, t0));
	ret.v4 = ADD64(ret.v4, MUL32(x.v0, t1));
	t0 = PERM64(y, SET02(3,3,3,3));
	ret.v0 = ADD64(ret.v0, MUL32(x.v0, t0));
	ret.v1 = ADD64(ret.v1, MUL32(x.v1, t0));
	ret.v2 = ADD64(ret.v2, MUL32(x.v2, t0));
	t0 = PERM32(t0, ord);
	ret.v3 = ADD64(ret.v3, MUL32(x.v2, t0));
	ret.v1 = ADD64(ret.v1, MUL32(x.v0, t0));
	ret.v2 = ADD64(ret.v2, MUL32(x.v1, t0));
	x.v1 = PERM32(x.v1, ord);
	ret.v0 = ADD64(ret.v0, MUL32(x.v1, t0));
	ret.v0 = ADD64(ret.v0, MUL32(x.v2, z));
	return ret;
}

static inline
vec256x5 mul4x130P(vec256x3 x, vec256 y, vec256 z) {
	vec256x5 ret;
	vec256 ord = SET32(6,7,4,5,2,3,0,1);
	ret.v0 = MUL32(x.v0, PERM32(y, SET64(0,0,0,0)));
	ret.v1 = MUL32(x.v0, PERM32(y, SET64(1,1,1,1)));
	ret.v2 = MUL32(x.v0, PERM32(y, SET64(2,2,2,2)));
	ret.v3 = MUL32(x.v0, PERM32(y, SET64(3,3,3,3)));
	ret.v4 = MUL32(x.v0, PERM32(y, SET64(4,4,4,4)));
	ret.v0 = ADD64(ret.v0, MUL32(x.v1, PERM32(y, SET64(7,7,7,7))));
	ret.v1 = ADD64(ret.v1, MUL32(x.v1, PERM32(y, SET64(0,0,0,0))));
	ret.v2 = ADD64(ret.v2, MUL32(x.v1, PERM32(y, SET64(1,1,1,1))));
	ret.v3 = ADD64(ret.v3, MUL32(x.v1, PERM32(y, SET64(2,2,2,2))));
	ret.v4 = ADD64(ret.v4, MUL32(x.v1, PERM32(y, SET64(3,3,3,3))));
	x.v0 = PERM32(x.v0, ord);
	x.v1 = PERM32(x.v1, ord);
	ret.v0 = ADD64(ret.v0, MUL32(x.v0, PERM32(y, SET64(6,6,6,6))));
	ret.v1 = ADD64(ret.v1, MUL32(x.v0, PERM32(y, SET64(7,7,7,7))));
	ret.v2 = ADD64(ret.v2, MUL32(x.v0, PERM32(y, SET64(0,0,0,0))));
	ret.v3 = ADD64(ret.v3, MUL32(x.v0, PERM32(y, SET64(1,1,1,1))));
	ret.v4 = ADD64(ret.v4, MUL32(x.v0, PERM32(y, SET64(2,2,2,2))));
	ret.v0 = ADD64(ret.v0, MUL32(x.v1, PERM32(y, SET64(5,5,5,5))));
	ret.v1 = ADD64(ret.v1, MUL32(x.v1, PERM32(y, SET64(6,6,6,6))));
	ret.v2 = ADD64(ret.v2, MUL32(x.v1, PERM32(y, SET64(7,7,7,7))));
	ret.v3 = ADD64(ret.v3, MUL32(x.v1, PERM32(y, SET64(0,0,0,0))));
	ret.v4 = ADD64(ret.v4, MUL32(x.v1, PERM32(y, SET64(1,1,1,1))));
	ret.v0 = ADD64(ret.v0, MUL32(x.v2, z));
	ret.v1 = ADD64(ret.v1, MUL32(x.v2, PERM32(y, SET64(5,5,5,5))));
	ret.v2 = ADD64(ret.v2, MUL32(x.v2, PERM32(y, SET64(6,6,6,6))));
	ret.v3 = ADD64(ret.v3, MUL32(x.v2, PERM32(y, SET64(7,7,7,7))));
	ret.v4 = ADD64(ret.v4, MUL32(x.v2, PERM32(y, SET64(0,0,0,0))));
	return ret;
}
static inline
vec256x5 muladd4x130P(vec256x5 ret, vec256x3 x, vec256 y, vec256 z) {
	vec256 ord = SET32(6,7,4,5,2,3,0,1);
	ret.v0 = ADD64(ret.v0, MUL32(x.v0, PERM32(y, SET64(0,0,0,0))));
	ret.v1 = ADD64(ret.v1, MUL32(x.v0, PERM32(y, SET64(1,1,1,1))));
	ret.v2 = ADD64(ret.v2, MUL32(x.v0, PERM32(y, SET64(2,2,2,2))));
	ret.v3 = ADD64(ret.v3, MUL32(x.v0, PERM32(y, SET64(3,3,3,3))));
	ret.v4 = ADD64(ret.v4, MUL32(x.v0, PERM32(y, SET64(4,4,4,4))));
	ret.v0 = ADD64(ret.v0, MUL32(x.v1, PERM32(y, SET64(7,7,7,7))));
	ret.v1 = ADD64(ret.v1, MUL32(x.v1, PERM32(y, SET64(0,0,0,0))));
	ret.v2 = ADD64(ret.v2, MUL32(x.v1, PERM32(y, SET64(1,1,1,1))));
	ret.v3 = ADD64(ret.v3, MUL32(x.v1, PERM32(y, SET64(2,2,2,2))));
	ret.v4 = ADD64(ret.v4, MUL32(x.v1, PERM32(y, SET64(3,3,3,3))));
	x.v0 = PERM32(x.v0, ord);
	x.v1 = PERM32(x.v1, ord);
	ret.v0 = ADD64(ret.v0, MUL32(x.v0, PERM32(y, SET64(6,6,6,6))));
	ret.v1 = ADD64(ret.v1, MUL32(x.v0, PERM32(y, SET64(7,7,7,7))));
	ret.v2 = ADD64(ret.v2, MUL32(x.v0, PERM32(y, SET64(0,0,0,0))));
	ret.v3 = ADD64(ret.v3, MUL32(x.v0, PERM32(y, SET64(1,1,1,1))));
	ret.v4 = ADD64(ret.v4, MUL32(x.v0, PERM32(y, SET64(2,2,2,2))));
	ret.v0 = ADD64(ret.v0, MUL32(x.v1, PERM32(y, SET64(5,5,5,5))));
	ret.v1 = ADD64(ret.v1, MUL32(x.v1, PERM32(y, SET64(6,6,6,6))));
	ret.v2 = ADD64(ret.v2, MUL32(x.v1, PERM32(y, SET64(7,7,7,7))));
	ret.v3 = ADD64(ret.v3, MUL32(x.v1, PERM32(y, SET64(0,0,0,0))));
	ret.v4 = ADD64(ret.v4, MUL32(x.v1, PERM32(y, SET64(1,1,1,1))));
	ret.v0 = ADD64(ret.v0, MUL32(x.v2, z));
	ret.v1 = ADD64(ret.v1, MUL32(x.v2, PERM32(y, SET64(5,5,5,5))));
	ret.v2 = ADD64(ret.v2, MUL32(x.v2, PERM32(y, SET64(6,6,6,6))));
	ret.v3 = ADD64(ret.v3, MUL32(x.v2, PERM32(y, SET64(7,7,7,7))));
	ret.v4 = ADD64(ret.v4, MUL32(x.v2, PERM32(y, SET64(0,0,0,0))));
	return ret;
}

/* functions for processing 2x 130 bits */
static inline
vec256x2 mul2x130(vec256x2 x, vec256 r1, vec256 r2, vec256 r15, vec256 r25) {
	vec256x2 ret;

	// multiply 2x2 and add both results simultaneously using lazy reduction, context switches from 32 bit to 64 bit
	ret.v0 = MUL32(PERM32(x.v0, SET64(4,3,2,1)), PERM32(r2, SET64(7,7,7,7)));
	ret.v1 = MUL32(PERM32(x.v1, SET64(4,3,2,1)), PERM32(r1, SET64(7,7,7,7)));
	ret.v0 = ADD64(ret.v0, MUL32(PERM64(x.v0, SET02(0,2,2,1)), PERM32(r2, SET64(3,6,5,6))));
	ret.v1 = ADD64(ret.v1, MUL32(PERM64(x.v1, SET02(0,2,2,1)), PERM32(r1, SET64(3,6,5,6)) ));
	ret.v0 = ADD64(ret.v0, MUL32(PERM32(x.v0, SET64(1,1,3,3)), PERM32(r2, SET64(2,1,6,5))));
	ret.v1 = ADD64(ret.v1, MUL32(PERM32(x.v1, SET64(1,1,3,3)), PERM32(r1, SET64(2,1,6,5))));
	ret.v0 = ADD64(ret.v0, MUL32(PERM32(x.v0, SET64(3,2,1,0)), BC32R(TC128(r2))));
	ret.v1 = ADD64(ret.v1, MUL32(PERM32(x.v1, SET64(3,2,1,0)), BC32R(TC128(r1))));
	vec256 t0 = PERM64(x.v0, SET02(1,0,0,2));
	vec256 t1 = PERM64(x.v1, SET02(1,0,0,2));
	ret.v0 = ADD64(ret.v0, MUL32(t0, MPLX(PERM32(r2, SET64(1,2,1,1)), r25, 0x03)));
	ret.v1 = ADD64(ret.v1, MUL32(t1, MPLX(PERM32(r1, SET64(1,2,1,1)), r15, 0x03)));
	ret.v0 = ADD64(ret.v0, ret.v1);
	t0 = MUL32(t0, r2);
	t1 = MUL32(t1, r1);
	ret.v1 = ADD64(t0, t1);
	t0 = MUL32(PERM32(x.v0, SET64(3,2,1,0)), PERM32(r2, SET64(1,2,3,4)));
	t1 = MUL32(PERM32(x.v1, SET64(3,2,1,0)), PERM32(r1, SET64(1,2,3,4)));
	t0  = ADD64(t0, t1);
	t0  = ADD64(t0, PERM64(t0, SET02(1,0,3,2)));
	t0  = ADD64(t0, PERM64(t0, SET02(2,3,0,1)));
	ret.v1 = ADD64(ret.v1, t0);

	return ret;
}

/* functions for processing 1x 130 bit */
static inline
vec256 red5x64(vec256x2 x) {
	// carry chain
	x.v0 = ADD64(
			AND256(x.v0, SET64(-1,0x3ffffff,0x3ffffff,0x3ffffff)),
			PERM64(SRV64(x.v0, SET64(64,26,26,26)), SET02(2,1,0,3)));
	x.v1 = ADD64(x.v1, PERM64(SRI64(x.v0, 26), SET02(2,1,0,3)));
	x.v0 = AND256 (x.v0, SET64(0x3ffffff,-1,-1,-1));

	// reduction modulus 2^130-5
	vec256 t = SRV64(x.v1, SET64(64,64,64,26));
	x.v0 = ADD64(ADD64(x.v0, t), SLI32(t, 2));
	x.v1 = AND256 (x.v1, SET64(0,0,0,0x3ffffff));

	// carry chain
	x.v0 = ADD64(
			AND256(x.v0, SET64(-1,0x3ffffff,0x3ffffff,0x3ffffff)),
			PERM64(SRV64(x.v0, SET64(64,26,26,26)), SET02(2,1,0,3)));
	x.v1 = ADD64(x.v1, PERM64(SRI64(x.v0, 26), SET02(2,1,0,3)));
	x.v0 = AND256 (x.v0, SET64(0x3ffffff,-1,-1,-1));

	// switch context from 64 bit to 32 bit
	return MPLX(
			PERM32(x.v0, SET32(0,6,4, 0,6,4,2,0)),
			PERM32(x.v1, SET32(0,6,4, 0,6,4,2,0)),
			0x90);
}
static inline
vec256x2 mul130(vec256 x, vec256 y, vec256 z) {
	vec256x2 ret;

	// multiply 2 values using lazy reduction, context switches from 32 bit to 64 bit
	ret.v0 = MUL32(PERM32(x, SET64(4,3,2,1)), PERM32(y, SET64(7,7,7,7)));
	ret.v0 = ADD64(ret.v0, MUL32(PERM32(x, SET64(3,2,1,0)), BC32R(TC128(y))));
	ret.v0 = ADD64(ret.v0, MUL32(PERM32(x, SET64(1,1,3,3)), PERM32(y, SET64(2,1,6,5))));
	ret.v0 = ADD64(ret.v0, MUL32(PERM64(x, SET02(1,0,0,2)), MPLX(PERM32(y, SET64(1,2,1,1)), z, 0x03)));
	ret.v0 = ADD64(ret.v0, MUL32(PERM64(x, SET02(0,2,2,1)), PERM32(y, SET64(3,6,5,6))));

	ret.v1  = MUL32(PERM32(x, SET64(3,2,1,0)), PERM32(y, SET64(1,2,3,4)));
	ret.v1 = ADD64(ret.v1, PERM64(ret.v1, SET02(1,0,3,2)));
	ret.v1  = ADD64(ret.v1, PERM64(ret.v1, SET02(0,0,0,1)));
	ret.v1 = ADD64(ret.v1, MUL32(PERM64(x, SET02(0,0,0,2)), y));

	return ret;
}
static inline
vec128 addkey(vec256 x, vec256 k) {
	vec256 t;
	// reduce modulus 2^130-5
	t = PERM32(SRI32(x, 26), SET32(7,7,7,3,2,1,0,4));
	x = ADD32(ADD32(
			AND256(x, SET32(0,0,0,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff,0x3ffffff)),
			t),
			PERM32(SLI32(t, 2), SET32(7,7,7,7,7,7,7,0)));

	// align back to 32 bits per digit
	x = OR256(
			SRV32(x, SET32(32,32,32,32,18,12,6,0)),
			PERM32(SLV32(x, SET32(32,32,32,8,14,20,26,32)), SET32(7,7,7,7,4,3,2,1)));

	// add key and store result in lower 128 bit
	x = ADD64(PERM32(x, SET32(7,3,7,2,7,1,7,0)), k);
	x = ADD64(
			PERM32(x, SET32(7,7,7,7,6,4,2,0)),
			PERM32(SRI64(x, 32), SET32(7,7,7,7,4,2,0,7)));

	// reduce modulus 2^128
	return TC128(x);
}

/* additional helper functions */
static inline
poly1305_state_internal_avx2* poly1305_aligned_state_avx2(poly1305_state_avx2 *state) {
	return (poly1305_state_internal_avx2 *)(((size_t)state + 63) & ~63);
}
static inline
void memcpy63(uint8_t *dst, const uint8_t *src, size_t bytes) {
	size_t offset = src - dst;
	if (bytes & 32) { STORE256(dst, LOAD256(dst + offset)); dst += 32; }
	if (bytes & 16) { STORE128(dst, LOAD128(dst + offset)); dst += 16; }
	if (bytes &  8) { *(uint64_t *)dst = *(uint64_t *)(dst + offset); dst += 8;}
	if (bytes &  4) { *(uint32_t *)dst = *(uint32_t *)(dst + offset); dst += 4;}
	if (bytes &  2) { *(uint16_t *)dst = *(uint16_t *)(dst + offset); dst += 2;}
	if (bytes &  1) { *( uint8_t *)dst = *( uint8_t *)(dst + offset); }

          
}
static inline
void memzero15(uint8_t *dst, size_t bytes) {
	if (bytes &  8) { *(uint64_t *)dst = 0; dst += 8; }
	if (bytes &  4) { *(uint32_t *)dst = 0; dst += 4; }
	if (bytes &  2) { *(uint16_t *)dst = 0; dst += 2; }
	if (bytes &  1) { *( uint8_t *)dst = 0; }
}

/* poly1305 interface functions */
void CRYPTO_poly1305_init_avx2(poly1305_state_avx2* state, const unsigned char key[32]) {
	poly1305_state_internal_avx2 *st = poly1305_aligned_state_avx2(state);
	vec256 k = LOAD256(key);

	// prepare and save addition key to state buffer
	st->k = AND256(PERM32(k, SET32(3,7,2,6,1,5,0,4)), SET32(0,-1,0,-1,0,-1,0,-1));

	// prepare polynomial key r
	vec256 r1 = ALIGN26(AND256(k, SET32(0,0,0,0,0x0ffffffc,0x0ffffffc,0x0ffffffc,0x0fffffff)));

	// precompute 5*r, r^2, 5*r^2
	vec256 r15 = PERM32(ADD32(r1, SLI32(r1, 2)), SET32(4,3,2,1,1,1,1,1));
	r1 = MPLX(r1, r15, 0xE0); r15 = BC128P(r15);
	vec256 r2 = red5x64(mul130(r1, r1, r15));
	vec256 r25 = PERM32(ADD32(r2, SLI32(r2, 2)), SET32(4,3,2,1,1,1,1,1));
	r2 = MPLX(r2, r25, 0xE0); r25 = BC128P(r25);

	// save polynomial keys r, 5*r, r^2, 5*r^2 to state buffer
	st->r1 = r1; st->r2 = r2; st->r15 = r15; st->r25 = r25;

	// clear initialization flag
	st->p_init = 0;

	// clear buffer length
	st->leftover = 0;

	return;
}


void CRYPTO_poly1305_modified_update_avx2(poly1305_state_avx2* state, const unsigned char *in, size_t N,unsigned char mac[16]) {

    size_t flag=0;//,last=0;
    int rem,in_len;

    /*modifying the input length to logically prepend zeroes*/
    in_len=N+64-((N%64));
    /********************************************************/
    /*computing the input type*/
    if(N%16!=0){
		flag=1;
    		rem=((N/16)+1)%4;
    }
	else
		rem=(N/16)%4;
    /********************************************************/

   	poly1305_state_internal_avx2 *st = poly1305_aligned_state_avx2(state);
	uint32_t *ip=(uint32_t *)in;
	vec256x3 p=p;
        //

	// precompute and save: r^3, r^4, 5*r^4 to state buffer               
	vec256 r1 = st->r1, r15 = st->r15, r2 = st->r2, r25 = st->r25;
	vec256 r3 = red5x64(mul130(r2, r1, r15));
	vec256 r4 = red5x64(mul130(r2, r2, r25));
	st->m.v0 = MPLX(r3, PERM32(r2, SET32(4,3,1,0,0,0,0,0)), 0xE0);
	st->m.v1 = MPLX(r4, PERM32(r2, SET32(4,2,0,0,0,0,0,0)), 0xE0);
	vec256 r45 = PERM32(ADD32(r4, SLI32(r4, 2)), SET32(4,3,2,1,1,1,1,1));
	st->r4 = MPLX(r4, r45, 0xE0);
	st->r45 = BC128P(r45);
			
			/*preparations for first load*/	
		if(in_len==64)
			p = align4x128_last(load4x128_test1(ip,16-N%16),16-N%16);
           	else 
		{	
                        if(N>=832)
				p = align4x128_large(load4x128_test(ip,rem),rem);
                        else
                                p = align4x128_init(load4x128_test(ip,rem),rem); 

              		if(rem!=0)
			  ip=ip+(rem*4);
		  	else
			  ip=ip+16;
            	}        
              	in_len=in_len-64;		
		// update polynomial with remaining input data

		  if (N>=832) {//for messages with length >= 832 bytes
                 
			// precompute r^8, 5*r^8, r^12, 5*r^12
			vec256 r8 = red5x64(mul130(st->r4, st->r4, st->r45));
			vec256 r12 = red5x64(mul130(r8, st->r4, st->r45));
			vec256 r85 = PERM32(ADD32(r8, SLI32(r8, 2)), SET32(4,3,2,1,1,1,1,1));
			vec256 r125 = PERM32(ADD32(r12, SLI32(r12, 2)), SET32(4,3,2,1,1,1,1,1));
			r8 = MPLX(r8, r85, 0xE0);
			r12 = MPLX(r12, r125, 0xE0);
			r85 = BC128P(r85);
			r125 = BC128P(r125);
                        
			do {

                                   if(in_len==192 && flag==1){
                                	
                                        p = add4x130(
						red4x130(muladd4x130P(muladd4x130P(
								mul4x130P(p, r12, r125),
								align4x128(load4x128(ip)), r8, r85),
								align4x128(load4x128(ip+16)), st->r4, st->r45)),
								align4x128_last(load4x128_test1(ip+32,16-N%16),16-N%16));
              		        }

				else{
                                            p = add4x130(
						red4x130(muladd4x130P(muladd4x130P(
								mul4x130P(p, r12, r125),
								align4x128(load4x128(ip)), r8, r85),
								align4x128(load4x128(ip+16)), st->r4, st->r45)),
								align4x128(load4x128(ip+32)));
                		}
				in_len -= 192; ip += 48;
			} while (in_len >= 192);

                        // compute 128 byte block (remaining input data < 192 bytes)
			if (in_len >= 128) {
                                if(in_len==128 && flag==1){
				p = add4x130(
						red4x130(muladd4x130P(
								mul4x130P(p, r8, r85),
								align4x128(load4x128(ip)), st->r4, st->r45)),
								align4x128_last(load4x128_test1(ip+16,16-N%16),16-N%16));
               			 }
                		else{
					p = add4x130(
						red4x130(muladd4x130P(
								mul4x130P(p, r8, r85),
								align4x128(load4x128(ip)), st->r4, st->r45)),
								align4x128(load4x128(ip+16)));
                
				}
				in_len -= 128; ip += 32;
			}

			// compute 64 byte block (remaining input data < 128 bytes)
		        if (in_len >= 64) {
                                if(in_len==64 && flag==1){
				p = add4x130(
						red4x130(mul4x130P(p, st->r4, st->r45)),
						align4x128_last(load4x128_test1(ip,16-N%16),16-N%16));
                		}
                		else{
					p = add4x130(
						red4x130(mul4x130P(p, st->r4, st->r45)),
						align4x128(load4x128(ip)));

				}
				in_len -= 64; ip += 16;

			}
		}

		// update polynomial with remaining message length...for messages with length < 832 bytes
		else {

			if(rem==1){//case 1:remainder=1...first simd multiplication using mul4x130R_rem1 module
					if(flag==1 && in_len==64)	       
						p = add4x130(
						red4x130(mul4x130R_rem1(p, st->r4, st->r45)),
						align4x128_last(load4x128_test1(ip,16-N%16),16-N%16));
					else
						p = add4x130(
						red4x130(mul4x130R_rem1(p, st->r4, st->r45)),
						align4x128(load4x128(ip)));
				
				in_len -= 64; ip += 16;
			}
	     		//case 2:remainder=2...first simd multiplication using mul4x130R_rem2 module	
             		else if(rem==2){
					if(flag==1 && in_len==64)	       
						p = add4x130(
						red4x130(mul4x130R_rem2(p, st->r4, st->r45)),
						align4x128_last(load4x128_test1(ip,16-N%16),16-N%16));
					else
						p = add4x130(
						red4x130(mul4x130R_rem2(p, st->r4, st->r45)),
						align4x128(load4x128(ip)));
				
				in_len -= 64; ip += 16;
			}   

			while (in_len >= 64)  {
                	 if(in_len==64 && flag==1){
					p = add4x130(
						red4x130(mul4x130R(p, st->r4, st->r45)),
						align4x128_last(load4x128_test1(ip,16-N%16),16-N%16));
				
                	}
                 	else{
					p = add4x130(
						red4x130(mul4x130R(p, st->r4, st->r45)),
						align4x128(load4x128(ip)));
		 	 }
			in_len -= 64; ip += 16;
		       }
		}

        	vec256 p1=p1;		
       		p1 = red5x64(hadd4x130(mul4x130M(p, st->m, st->r1)));

		/********************************************************* END OF MULTIPLICATIONS USING 4-DECIMATION HORNER ************************************************************/
        	

    		STORE128(mac, addkey(p1, st->k));
	
	
	return;
}

/*void CRYPTO_poly1305_modified_finish_avx2(poly1305_state_avx2* state, unsigned char mac[16]) {

	poly1305_state_internal_avx2 *st = poly1305_aligned_state_avx2(state);
	
	vec256 p=p;

    p = red5x64(hadd4x130(mul4x130M(st->p, st->m, st->r1)));
	

    STORE128(mac, addkey(p, st->k));

	
}*/


void CRYPTO_poly1305_update_avx2(poly1305_state_avx2* state, const unsigned char *in, size_t in_len,unsigned char mac[16]) {
	poly1305_state_internal_avx2 *st = poly1305_aligned_state_avx2(state);
	uint32_t *ip=(uint32_t *)in;

	//if ((st->leftover + in_len) >= 64) {
		vec256x3 p=p;
		//if (!st->p_init)  {
			// precompute and save: r^3, r^4, 5*r^4 to state buffer
			vec256 r1 = st->r1, r15 = st->r15, r2 = st->r2, r25 = st->r25;
			vec256 r3 = red5x64(mul130(r2, r1, r15));
			vec256 r4 = red5x64(mul130(r2, r2, r25));
			st->m.v0 = MPLX(r3, PERM32(r2, SET32(4,3,1,0,0,0,0,0)), 0xE0);
			st->m.v1 = MPLX(r4, PERM32(r2, SET32(4,2,0,0,0,0,0,0)), 0xE0);
			vec256 r45 = PERM32(ADD32(r4, SLI32(r4, 2)), SET32(4,3,2,1,1,1,1,1));
			st->r4 = MPLX(r4, r45, 0xE0);
			st->r45 = BC128P(r45);

			// initialize polynomial p
		/*	if (st->leftover) {
				uint32_t len = 64 - st->leftover;
				memcpy63(&st->buffer.b[st->leftover], (uint8_t *)ip, len);
				in_len -= len; ip = (uint32_t *)((uint8_t *)ip + len);
				p = align4x128(st->buffer.v);
				st->leftover = 0;
			}*/
			//else {
				p = align4x128(load4x128(ip));
				in_len -= 64; ip += 16;
			//}

			// set initialization flag
			st->p_init = 1;
		//}
		/*else if (st->leftover) {
			// update polynomial with saved data from last call
			uint32_t len = 64 - st->leftover;
			memcpy63(&st->buffer.b[st->leftover], (uint8_t *)ip, len);
			in_len -= len; ip = (uint32_t *)((uint8_t *)ip + len);
			p = add4x130(
					red4x130(mul4x130R(st->p, st->r4, st->r45)),
					align4x128(st->buffer.v));
			st->leftover = 0;
		}
		// load polynomial from state buffer
		else p = st->p;*/

		// update polynomial with input data
		if (in_len >= 768) {
			// precompute r^8, 5*r^8, r^12, 5*r^12
			vec256 r8 = red5x64(mul130(st->r4, st->r4, st->r45));
			vec256 r12 = red5x64(mul130(r8, st->r4, st->r45));
			vec256 r85 = PERM32(ADD32(r8, SLI32(r8, 2)), SET32(4,3,2,1,1,1,1,1));
			vec256 r125 = PERM32(ADD32(r12, SLI32(r12, 2)), SET32(4,3,2,1,1,1,1,1));
			r8 = MPLX(r8, r85, 0xE0);
			r12 = MPLX(r12, r125, 0xE0);
			r85 = BC128P(r85);
			r125 = BC128P(r125);

			// update polynomial in 192 byte blocks
			do {
				p = add4x130(
						red4x130(muladd4x130P(muladd4x130P(
								mul4x130P(p, r12, r125),
								align4x128(load4x128(ip)), r8, r85),
								align4x128(load4x128(ip+16)), st->r4, st->r45)),
								align4x128(load4x128(ip+32)));
				in_len -= 192; ip += 48;
			} while (in_len >= 192);


			if (in_len >= 128) {
				// compute 128 byte block (remaining input data < 192 bytes)
				p = add4x130(
						red4x130(muladd4x130P(
								mul4x130P(p, r8, r85),
								align4x128(load4x128(ip)), st->r4, st->r45)),
								align4x128(load4x128(ip+16)));
				in_len -= 128; ip += 32;
			}
			else if (in_len >= 64) {
				// compute 64 byte block (remaining input data < 128 bytes)
				p = add4x130(
						red4x130(mul4x130P(p, st->r4, st->r45)),
						align4x128(load4x128(ip)));
				in_len -= 64; ip += 16;
			}
		}
		else if (in_len >= 64) {
			// update polynomial in 64 byte blocks
			do  {
				p = add4x130(
						red4x130(mul4x130R(p, st->r4, st->r45)),
						align4x128(load4x128(ip)));
				in_len -= 64; ip += 16;
			} while (in_len >= 64);
		}

		// save updated polynomial to state buffer
		//st->p = p;
	//}
       
       //vec256 p1=p1;
      vec256 p1 = red5x64(hadd4x130(mul4x130M(p, st->m, st->r1))); 
       STORE128(mac, addkey(p1, st->k));

	return;
}

void CRYPTO_poly1305_finish_avx2(poly1305_state_avx2* state, const unsigned char *in, size_t in_len,unsigned char mac[16]) {
	poly1305_state_internal_avx2 *st = poly1305_aligned_state_avx2(state);
        uint32_t *ip=(uint32_t *)in;
	uint32_t p_init=st->p_init;
	vec256 p=p;
        memcpy63(&st->buffer.b[st->leftover], (uint8_t *)ip, in_len);
	st->leftover += in_len;

		uint32_t buf_len=st->leftover, idx=0;

		if (buf_len >= 32) {
			// compute 32 byte block (remaining data < 64 bytes)
			vec256x2 c = {
				ALIGN26(OR256(TC256(st->buffer.va[0]), SET64(0,1,0,0))),
				ALIGN26(OR256(TC256(st->buffer.va[1]), SET64(0,1,0,0)))
			};
			//if (p_init) c.v0 = ADD32(p, c.v0);
			p = red5x64(mul2x130(c, st->r1, st->r2, st->r15, st->r25));
			idx += 2; buf_len -= 32; p_init++;
		}

		if (buf_len >= 16) {
			// compute 16 byte block (remaining data < 32 bytes)
			vec256 c = ALIGN26(OR256(TC256(st->buffer.va[idx]), SET64(0,1,0,0)));
			if (p_init) c = ADD32(p, c);
			p = red5x64(mul130(c, st->r1, st->r15));
			idx++; buf_len -= 16; p_init++;
		}

		if (buf_len) {
			// compute last block (remaining data < 16 bytes)
			st->buffer.b[st->leftover] = 1;
			if (buf_len < 15) memzero15(&st->buffer.b[st->leftover+1], 15-buf_len);
			vec256 c = ALIGN26(TC256(st->buffer.va[idx]));
			if (p_init) c = ADD32(p, c);
			p = red5x64(mul130(c, st->r1, st->r15));
			p_init++;
		}
	

	// compute tag: p + k mod 2^128
	if (p_init) STORE128(mac, addkey(p, st->k));
	else 		
        STORE128(mac, TC128(PERM32(st->k, SET32(0,0,0,0,6,4,2,0))));

	return;
}


#else
#error -- Implementation supports only microarchitectures with Advanced Vector Extension 2 support
#endif

