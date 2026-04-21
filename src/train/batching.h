#pragma once

#include <ATen/ops/randint.h>
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <stdexcept>
#include <vector>

namespace train {

struct Batch {
    torch::Tensor x; // [B, T]
    torch::Tensor y; // [B, T]
};

inline Batch get_batch(const torch::Tensor& tokens,
                       int64_t batch_size,
                       int64_t context_len,
                       torch::Device device) {

    if (tokens.dim() != 1) {
        throw std::runtime_error("get_batch: tokens must be a 1D tensor.");
    }

    const auto N = tokens.size(0);
    if (N <= context_len + 1) {
        throw std::runtime_error("get_batch: token stream too short.");
    }

    const auto max_start = N - context_len - 1;
    auto starts = torch::randint(
        /*high=*/max_start,
        /*size=*/{batch_size},
        torch::TensorOptions().dtype(torch::kLong));


    std::vector<torch::Tensor> xs;
    std::vector<torch::Tensor> ys;
    xs.reserve(batch_size);
    ys.reserve(batch_size);

    for (int64_t i = 0; i < batch_size; i++) {
        const auto s = starts[i].item<int64_t>();
        xs.push_back(tokens.slice(/*dim=*/0, /*start=*/s, /*end=*/s + context_len));
        ys.push_back(tokens.slice(/*dim=*/0, /*start=*/s + 1, /*end=*/s + 1 + context_len));
    }

    auto x = torch::stack(xs).to(device);
    auto y = torch::stack(ys).to(device);

    return {x, y};
}

} // namespace train
