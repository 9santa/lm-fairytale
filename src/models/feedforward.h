#pragma once

#include <torch/torch.h>

namespace models {

struct FeedForwardImpl : torch::nn::Module {
    FeedForwardImpl(int64_t d_model,
                    int64_t hidden_dim,
                    double dropout = 0.0)
        : net_(register_module(
        "net",
        torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(d_model, hidden_dim)),
            torch::nn::GELU(),
            torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, d_model)),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout))
        )
    )) {}

    torch::Tensor forward(const torch::Tensor& x) {
        // x: [B, T, d_model]
        return net_->forward(x);
    }

private:
    torch::nn::Sequential net_{nullptr};
};

TORCH_MODULE(FeedForward);


} // namespace models
