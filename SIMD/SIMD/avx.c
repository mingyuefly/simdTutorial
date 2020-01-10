//
//  avx.c
//  SIMD
//
//  Created by Gguomingyue on 2020/1/9.
//  Copyright © 2020 Gmingyue. All rights reserved.
//

#include "avx.h"
#include <immintrin.h>
#include <avxintrin.h>

double AVXAdd(double * input1, int length)
{
    int offset = 0;
    __m256d v1;
    __m256d sum = _mm256_setzero_pd();
    double ret = 0;
    for (int i = 0; i < length / 4; i++) {
        v1 = _mm256_load_pd(input1 + offset);
        sum = _mm256_add_pd(sum, v1);
        offset += 4;
    }
    sum = _mm256_hadd_pd(sum, sum);
    ret = sum[0] + sum[2];
    return ret;
}
