//
//  main.c
//  SIMD
//
//  Created by Gguomingyue on 2020/1/9.
//  Copyright © 2020 Gmingyue. All rights reserved.
//

#include <stdio.h>
#include <emmintrin.h>
#include <sys/time.h>
#include "avx.h"
#include "avx512simd.h"

void logArray(char *arrayName, double * logArray, int length);
double commonAdd(double * input1, int length);
double SSEAdd(double * input1, int length);

int main(int argc, const char * argv[]) {
    // 基础用法实例，向量相加
    const int k = 5;
    double input1[k] = {1, 2, 3, 4, 5};
    double input2[k] = {6, 7, 8, 9, 10};
    double result[k] = {0};
    double result2[k] = {0};
    for (int i = 0; i < k; i++) {
        result[i] = input1[i] + input2[i];
    }
    logArray("input1", input1, k);
    logArray("input2", input2, k);
    logArray("result", result, k);
    
    int limit = 0;
    if ((k % 2) == 1) {
        limit = k / 2 + 1;
    } else {
        limit = k / 2;
    }
    for (int i = 0; i < limit; i++) {
        __m128d a = _mm_load_pd(input1 + 2 * i);
        __m128d b = _mm_load_pd(input2 + 2 * i);
        __m128d c = _mm_add_pd(a, b);
        _mm_store_pd(result2 + 2 * i, c);
    }
    logArray("result2", result2, k);
    
    // 性能比较测试，加法求和
    const int k2 = 512 * 512;
    //const int k2 = 1024 * 1024;
    const int loop = 1;
    double input3[k2];
    for (int i = 0; i < k2; i++) {
        input3[i] = i;
    }
    
    struct  timeval   start;
    struct  timeval   end;
    
    double commonAddResult = 0;
    gettimeofday(&start, NULL);
    for (int j = 0; j < loop; j++) {
        commonAddResult += commonAdd(input3, k2);
    }
    gettimeofday(&end, NULL);
    printf("tv_sec:%ld\n",end.tv_sec - start.tv_sec);
    printf("tv_usec:%d\n", end.tv_usec - start.tv_usec);

    gettimeofday(&start, NULL);
    double SSEAddResult = 0;
    for (int j = 0; j < loop; j++) {
        SSEAddResult += SSEAdd(input3, k2);
    }
    gettimeofday(&end, NULL);
    printf("tv_sec:%ld\n",end.tv_sec - start.tv_sec);
    printf("tv_usec:%d\n", end.tv_usec - start.tv_usec);
    
    gettimeofday(&start, NULL);
    double AVXAddResult = 0;
    for (int j = 0; j < loop; j++) {
        AVXAddResult += AVXAdd(input3, k2);
    }
    gettimeofday(&end, NULL);
    printf("tv_sec:%ld\n",end.tv_sec - start.tv_sec);
    printf("tv_usec:%d\n", end.tv_usec - start.tv_usec);
    
    gettimeofday(&start, NULL);
    double AVX512AddResult = 0;
    for (int j = 0; j < loop; j++) {
        // avx512调用_mm512_setzero_pd函数会崩溃，具体发现是无法初始化__m512d变量，暂时没找到原因和解决方法
        //AVX512AddResult += AVX512Add(input3, k2);
    }
    gettimeofday(&end, NULL);
    printf("tv_sec:%ld\n",end.tv_sec - start.tv_sec);
    printf("tv_usec:%d\n", end.tv_usec - start.tv_usec);
    
    if (commonAddResult == SSEAddResult) {
        printf("correct\n");
    } else {
        printf("incorrect\n");
    }
    
    if (commonAddResult == AVXAddResult) {
        printf("correct1\n");
    } else {
        printf("incorrect1\n");
    }
    
    if (AVX512AddResult == AVXAddResult) {
        printf("correct2\n");
    } else {
        printf("incorrect2\n");
    }
    
    return 0;
}

void logArray(char *arrayName, double * logArray, int length)
{
    for (int i = 0; i < length; i++) {
        printf("%s[%d] = %.2f\n",arrayName, i, logArray[i]);
    }
}

double commonAdd(double * input1, int length)
{
    double result = 0;
    for (int i = 0; i < length; i++) {
        result += input1[i];
    }
    return result;
}

double SSEAdd(double * input1, int length)
{
    int offset = 0;
    __m128d v1;
    __m128d sum = _mm_setzero_pd();
    double ret = 0;
    for (int i = 0; i < length / 2; i++) {
        v1 = _mm_load_pd(input1 + offset);
        sum = _mm_add_pd(sum, v1);
        offset += 2;
    }
    ret = sum[0] + sum[1];
    return ret;
}
