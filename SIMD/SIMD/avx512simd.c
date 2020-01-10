//
//  avx512simd.c
//  SIMD
//
//  Created by Gguomingyue on 2020/1/9.
//  Copyright © 2020 Gmingyue. All rights reserved.
//

#include "avx512simd.h"
#include <immintrin.h>
#include <avx512fintrin.h>

double AVX512Add(double * input1, int length)
{
    int offset = 0;
    __m512d v1;
    __m512d sum = _mm512_setzero_pd();
    
    double ret = 0;
    for (int i = 0; i < length / 8; i++) {
        v1 = _mm512_load_pd(input1 + offset);
        sum = _mm512_add_pd(sum, v1);
        offset += 8;
    }
    //sum = _mm512_hadd_pd(sum, sum);

    ret = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
    return ret;
}
