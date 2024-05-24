#include "cpu_types.hpp"

namespace {

template <typename scalar_t>
void scaled_fp8_quant(torch::Tensor& out,
                      torch::Tensor& input,
                      torch::Tensor& scale) {
    int64_t num_tokens = input.numel() / input.size(-1);
    int64_t num_elems = input.numel();

    for (int64_t i = 0; i < num_tokens; ++i) {
        for (int64_t j = 0; j < num_elems; ++j) {
            out[i * num_elems + j] = static_cast<c10::Float8_e4m3fn>(input[i * num_elems + j]) * scale.item<float>();
        }
    }
}

}  // namespace


void static_scaled_fp8_quant(torch::Tensor& output, torch::Tensor& input,
                        torch::Tensor& scale) {
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "scaled_fp8_quant_kernel", [&] {
        scaled_fp8_quant<scalar_t>(out, input, scale);
    });

}
