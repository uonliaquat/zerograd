// file: gemm_blis.c
#include <stdio.h>
#include "blis.h"

int main(void) {
    bli_init();  // once per process

    obj_t A, B, C, alpha, beta;
    dim_t M=2, N=3, K=4;

    bli_obj_create_1x1( BLIS_DOUBLE, &alpha );
    bli_obj_create_1x1( BLIS_DOUBLE, &beta );
    bli_setsc( 1.0, 0.0, &alpha );   // alpha = 1.0
    bli_setsc( 0.0, 0.0, &beta  );   // beta  = 0.0

    bli_obj_create( BLIS_DOUBLE, M, K, 0, 0, &A );
    bli_obj_create( BLIS_DOUBLE, K, N, 0, 0, &B );
    bli_obj_create( BLIS_DOUBLE, M, N, 0, 0, &C );

    // Fill A, B
    double* pa = bli_obj_buffer_at_off( &A );
    double* pb = bli_obj_buffer_at_off( &B );
    for (int i=0;i<M*K;i++) pa[i] = i+1;            // simple data
    for (int i=0;i<K*N;i++) pb[i] = i+1;

    // C := alpha*A*B + beta*C
    bli_gemm( &alpha, &A, &B, &beta, &C );

    double* pc = bli_obj_buffer_at_off( &C );
    for (int i=0;i<M*N;i++) printf("%.1f%c", pc[i], (i%N==N-1)?'\n':' ');

    bli_obj_free( &A ); bli_obj_free( &B ); bli_obj_free( &C );
    bli_obj_free( &alpha ); bli_obj_free( &beta );
    bli_finalize();
    return 0;
}
