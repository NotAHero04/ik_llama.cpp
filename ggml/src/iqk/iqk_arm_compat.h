#include <arm_neon.h>

#ifndef __ARM_FEATURE_DOTPROD
static inline int32x4_t ggml_vdotq_s32(int32x4_t r, int8x16_t a, int8x16_t b) {
    const int16x8_t p_lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t p_hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    const int32x4_t s_lo = vpaddlq_s16(p_lo);
    const int32x4_t s_hi = vpaddlq_s16(p_hi);
    const int32x4_t dots = vpaddq_s32(s_lo, s_hi);
    return vaddq_s32(r, dots);
}
static inline int32x4_t ggml_vdotq_laneq_s32(int32x4_t r, int8x16_t a, int8x16_t b, const int lane) {
    const int32_t b_lane_val = ((const int32_t*)(&b))[lane];
    const int8x16_t b_bcast = vreinterpretq_s8_s32(vdupq_n_s32(b_lane_val));
    return ggml_vdotq_s32(r, a, b_bcast);
}
#else
#define ggml_vdotq_s32(r, a, b) vdotq_s32(r, a, b)
#define ggml_vdotq_laneq_s32(r, a, b, lane) vdotq_laneq_s32(r, a, b, lane)
#endif

#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static inline uint16x8_t ggml_vceqq_f16(float16x8_t a, float16x8_t b) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    uint32x4_t cmp_lo = vceqq_f32(a_lo, b_lo);
    uint32x4_t cmp_hi = vceqq_f32(a_hi, b_hi);
    uint16x4_t res_lo = vshrn_n_u32(cmp_lo, 16);
    uint16x4_t res_hi = vshrn_n_u32(cmp_hi, 16);
    return vcombine_u16(res_lo, res_hi);
}
static inline float16x8_t ggml_vmulq_f16(float16x8_t a, float16x8_t b) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    float32x4_t res_lo = vmulq_f32(a_lo, b_lo);
    float32x4_t res_hi = vmulq_f32(a_hi, b_hi);
    float16x4_t res_lo_f16 = vcvt_f16_f32(res_lo);
    float16x4_t res_hi_f16 = vcvt_f16_f32(res_hi);
    return vcombine_f16(res_lo_f16, res_hi_f16);
}
static inline float16x8_t ggml_vcvtq_f16_s16(int16x8_t a) {
    int32x4_t a_lo = vmovl_s16(vget_low_s16(a));
    int32x4_t a_hi = vmovl_s16(vget_high_s16(a));
    float32x4_t f_lo = vcvtq_f32_s32(a_lo);
    float32x4_t f_hi = vcvtq_f32_s32(a_hi);
    return vcombine_f16(vcvt_f16_f32(f_lo), vcvt_f16_f32(f_hi));
}
static inline float16x8_t ggml_vfmaq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    float32x4_t c_lo = vcvt_f32_f16(vget_low_f16(c));
    float32x4_t c_hi = vcvt_f32_f16(vget_high_f16(c));
    float32x4_t res_lo = vfmaq_f32(a_lo, b_lo, c_lo);
    float32x4_t res_hi = vfmaq_f32(a_hi, b_hi, c_hi);
    float16x4_t res_lo_f16 = vcvt_f16_f32(res_lo);
    float16x4_t res_hi_f16 = vcvt_f16_f32(res_hi);
    return vcombine_f16(res_lo_f16, res_hi_f16);
}
/* Clang isn't happy with this
static inline float16x8_t ggml_vfmaq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c, const int lane) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    float32x4_t c_f32 = vcvt_f32_f16(c);
    float32x4_t c_lane_bcast = vdupq_n_f32(vgetq_lane_f32(c_f32, lane));
    float32x4_t res_lo = vfmaq_f32(a_lo, b_lo, c_lane_bcast);
    float32x4_t res_hi = vfmaq_f32(a_hi, b_hi, c_lane_bcast);
    return vcombine_f16(vcvt_f16_f32(res_lo), vcvt_f16_f32(res_hi));
}
*/
static inline float16x8_t ggml_vfmaq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c, const int lane) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));
    float32x4_t c_f32 = vcvt_f32_f16(c);
    float32_t c_array[4];
    vst1q_f32(c_array, c_f32);
    float32_t c_scalar = c_array[lane];
    float32x4_t res_lo = vmlaq_n_f32(a_lo, b_lo, c_scalar);
    float32x4_t res_hi = vmlaq_n_f32(a_hi, b_hi, c_scalar);
    return vcombine_f16(vcvt_f16_f32(res_lo), vcvt_f16_f32(res_hi));
}
static inline float16x8_t ggml_vaddq_f16(float16x8_t a, float16x8_t b) {
    float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b));

    float32x4_t res_lo = vaddq_f32(a_lo, b_lo);
    float32x4_t res_hi = vaddq_f32(a_hi, b_hi);

    float16x4_t res_lo_f16 = vcvt_f16_f32(res_lo);
    float16x4_t res_hi_f16 = vcvt_f16_f32(res_hi);

    return vcombine_f16(res_lo_f16, res_hi_f16);
}
static inline float16x4_t ggml_vadd_f16(float16x4_t a, float16x4_t b) {
    float32x4_t a_f32 = vcvt_f32_f16(a);
    float32x4_t b_f32 = vcvt_f32_f16(b);
    float32x4_t res_f32 = vaddq_f32(a_f32, b_f32);
    return vcvt_f16_f32(res_f32);
}
static inline float16x8_t ggml_vpaddq_f16(float16x8_t a, float16x8_t b) {
    float32x4_t a_lo_f32 = vcvt_f32_f16(vget_low_f16(a));
    float32x4_t a_hi_f32 = vcvt_f32_f16(vget_high_f16(a));
    float32x4_t b_lo_f32 = vcvt_f32_f16(vget_low_f16(b));
    float32x4_t b_hi_f32 = vcvt_f32_f16(vget_high_f16(b));
    float32x4_t res_lo_f32 = vpaddq_f32(a_lo_f32, b_lo_f32);
    float32x4_t res_hi_f32 = vpaddq_f32(a_hi_f32, b_hi_f32);
    float16x4_t res_lo_f16 = vcvt_f16_f32(res_lo_f32);
    float16x4_t res_hi_f16 = vcvt_f16_f32(res_hi_f32);
    return vcombine_f16(res_lo_f16, res_hi_f16);
}
static inline float16x8_t ggml_vabsq_f16(float16x8_t a) {
    return vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(a), vdupq_n_u16(0x8000)));
}
static inline float16x4_t ggml_vmul_f16(float16x4_t a, float16x4_t b) {
    float32x4_t a_f32 = vcvt_f32_f16(a);
    float32x4_t b_f32 = vcvt_f32_f16(b);
    float32x4_t res_f32 = vmulq_f32(a_f32, b_f32);
    return vcvt_f16_f32(res_f32);
}
#else
#define ggml_vceqq_f16(a, b) vceqq_f16(a, b)
#define ggml_vmulq_f16(a, b) vmulq_f16(a, b)
#define ggml_vcvtq_f16_s16(a) vcvtq_f16_s16(a)
#define ggml_vfmaq_f16(a, b, c) vfmaq_f16(a, b, c)
#define ggml_vfmaq_lane_f16(a, b, c, lane) vfmaq_lane_f16(a, b, c, lane)
#define ggml_vaddq_f16(a, b) vaddq_f16(a, b)
#define ggml_vadd_f16(a, b) vadd_f16(a, b)
#define ggml_vpaddq_f16(a, b) vpaddq_f16(a, b)
#define ggml_vabsq_f16(a) vabsq_f16(a)
#define ggml_vmul_f16(a, b) vmul_f16(a, b)
#endif
