// Distance kernels.
//
// This is where nearly all the time goes. An HNSW search at ef=100 on a 10k
// index runs a few thousand distance computations, and everything else in the
// index is bookkeeping around them, so it's worth hand vectorizing.
//
// NEON on arm64, AVX on x86, scalar fallback everywhere else. The scalar path
// is not dead code, the tests check every path agrees to within float error.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define ANNLITE_NEON 1
#include <arm_neon.h>
#elif defined(__AVX__)
#define ANNLITE_AVX 1
#include <immintrin.h>
#endif

namespace annlite {

enum class Metric { L2, InnerProduct, Cosine };

inline float l2_scalar(const float* a, const float* b, size_t d) {
  float sum = 0.0f;
  for (size_t i = 0; i < d; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

inline float dot_scalar(const float* a, const float* b, size_t d) {
  float sum = 0.0f;
  for (size_t i = 0; i < d; ++i) sum += a[i] * b[i];
  return sum;
}

#if ANNLITE_NEON

// Four independent accumulators rather than one. The adds have ~3 cycle
// latency and the loop is latency bound, not throughput bound, so a single
// accumulator would serialize the whole thing on its own dependency chain.
inline float l2_neon(const float* a, const float* b, size_t d) {
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
  float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    s0 = vfmaq_f32(s0, d0, d0);
    s1 = vfmaq_f32(s1, d1, d1);
    s2 = vfmaq_f32(s2, d2, d2);
    s3 = vfmaq_f32(s3, d3, d3);
  }
  for (; i + 4 <= d; i += 4) {
    float32x4_t dv = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    s0 = vfmaq_f32(s0, dv, dv);
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < d; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

inline float dot_neon(const float* a, const float* b, size_t d) {
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
  float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= d; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < d; ++i) sum += a[i] * b[i];
  return sum;
}

#endif  // ANNLITE_NEON

#if ANNLITE_AVX

inline float l2_avx(const float* a, const float* b, size_t d) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    s0 = _mm256_add_ps(s0, _mm256_mul_ps(d0, d0));
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(d1, d1));
  }
  __m256 s = _mm256_add_ps(s0, s1);
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, s);
  float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < d; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

inline float dot_avx(const float* a, const float* b, size_t d) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
  }
  __m256 s = _mm256_add_ps(s0, s1);
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, s);
  float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < d; ++i) sum += a[i] * b[i];
  return sum;
}

#endif  // ANNLITE_AVX

inline float l2(const float* a, const float* b, size_t d) {
#if ANNLITE_NEON
  return l2_neon(a, b, d);
#elif ANNLITE_AVX
  return l2_avx(a, b, d);
#else
  return l2_scalar(a, b, d);
#endif
}

inline float dot(const float* a, const float* b, size_t d) {
#if ANNLITE_NEON
  return dot_neon(a, b, d);
#elif ANNLITE_AVX
  return dot_avx(a, b, d);
#else
  return dot_scalar(a, b, d);
#endif
}

// Everything downstream minimizes, so inner product and cosine get negated.
// Cosine assumes vectors were normalized on insert, which the index does, so
// this stays a single dot product instead of two norms per comparison.
inline float distance(const float* a, const float* b, size_t d, Metric m) {
  switch (m) {
    case Metric::L2:
      return l2(a, b, d);
    case Metric::InnerProduct:
    case Metric::Cosine:
      return -dot(a, b, d);
  }
  return 0.0f;
}

inline const char* simd_backend() {
#if ANNLITE_NEON
  return "neon";
#elif ANNLITE_AVX
  return "avx";
#else
  return "scalar";
#endif
}

inline void normalize(float* v, size_t d) {
  const float n = std::sqrt(dot(v, v, d));
  if (n > 0.0f) {
    const float inv = 1.0f / n;
    for (size_t i = 0; i < d; ++i) v[i] *= inv;
  }
}

}  // namespace annlite
