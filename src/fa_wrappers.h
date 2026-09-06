/*
 * fa_wrappers.h - internal per-stage host wrappers (declarations).
 */
#pragma once
#include "fa_api.h"

fa_status fa1_fwd_f32(const float* Q, const float* K, const float* V, float* O,
                      int N, int d, cudaStream_t stream);
fa_status fa2_fwd_f32(const float* Q, const float* K, const float* V, float* O,
                      int N, int d, cudaStream_t stream);
fa_status fa3_fwd_f16(const void* Q, const void* K, const void* V, float* O,
                      int N, int d, cudaStream_t stream);
fa_status fa4_fwd_f16(int variant,
                      const void* Q, const void* K, const void* V, float* O,
                      int N, int d, cudaStream_t stream);
