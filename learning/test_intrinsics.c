#include <stdio.h>
#include <xmmintrin.h>
#include <immintrin.h>
int main(){
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float c[8] = {0};

    __m256 vec_a = _mm256_loadu_ps(a);
    __m256 vec_b = _mm256_loadu_ps(b);
    __m256 vec_c = _mm256_add_ps(vec_a, vec_b);

    _mm256_storeu_ps(c, vec_c);

    for(int i = 0; i < 8; i++){
        printf("%f.2f\n", c[i]);
    }

    return 0;
}
