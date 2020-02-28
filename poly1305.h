#ifndef POLY1305_H
#define POLY1305_H

#include <stdint.h>


typedef unsigned char poly1305_state_avx2[576];





void CRYPTO_poly1305_init_avx2(poly1305_state_avx2 *state, const unsigned char key[32]);
void CRYPTO_poly1305_update_avx2(poly1305_state_avx2 *state, const unsigned char *m, size_t bytes,unsigned char mac[16]);
void CRYPTO_poly1305_finish_avx2(poly1305_state_avx2 *state, const unsigned char *m, size_t bytes,unsigned char mac[16]);
void CRYPTO_poly1305_modified_update_avx2(poly1305_state_avx2 *state, const unsigned char *m, size_t bytes,unsigned char mac[16]);



#endif
