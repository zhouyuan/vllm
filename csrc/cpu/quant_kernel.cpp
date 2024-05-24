#include "cpu_types.hpp"

namespace {

#define FP8_E4M3_MAX std::numeric_limits<c10::Float8_e4m3fn>::max()

template <typename scalar_t>
c10::Float8_e4m3fn scaled_fp8_conversion(const scalar_t val, const float scale) {
  float x = static_cast<float>(val) / scale;
  float r = std::fmax(-FP8_E4M3_MAX, std::fmin(x, FP8_E4M3_MAX));
  return static_cast<c10::Float8_e4m3fn>(r);
}

/*
template <typename scalar_t>
void scaled_fp8_quant_kernel(scalar_t* out, const scalar_t* input, float* scale, int64_t num_elems) {
    #pragma omp parallel for
    for (int64_t i = 0; i < num_elems; ++i) {
        out[i] = static_cast<scalar_t>(input[i] * scale[0]);
    }
}

template <typename scalar_t>
void static_scaled_fp8_quant(torch::Tensor& out, torch::Tensor& input, torch::Tensor& scale) {
    int64_t num_tokens = input.numel() / input.size(-1);
    int64_t num_elems = input.numel();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "scaled_fp8_quant", [&] {
        scaled_fp8_quant_kernel<scalar_t>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), scale.data_ptr<float>(), num_elems);
    });
}
*/
template <typename scalar_t>
void scaled_fp8_quant_kernel(c10::Float8_e4m3fn* out, const scalar_t* input, float* scale, int64_t num_elems) {
    #pragma omp parallel for
    for (int64_t i = 0; i < num_elems; ++i) {
        out[i] = scaled_fp8_conversion(input[i], scale[0]);
    }
}

}  // namespace


void static_scaled_fp8_quant(torch::Tensor& output, torch::Tensor& input,
                        torch::Tensor& scale) {
    int64_t num_elems = input.numel();

  //AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "scaled_fp8_quant_kernel", [&] {
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "scaled_fp8_quant_kernel", [&] {
        scaled_fp8_quant_kernel<scalar_t>(output.data_ptr<c10::Float8_e4m3fn>(), input.data_ptr<scalar_t>(), scale.data_ptr<float>(), num_elems);
    });

}
