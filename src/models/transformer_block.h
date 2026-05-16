#pragma once

#include <cstdint>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/options/normalization.h>
#include <torch/torch.h>

#include "models/self_attention.h"
#include "models/feedforward.h"

namespace models {

struct TransformerBlockImpl : torch::nn::Module {
    TransformerBlockImpl(int64_t d_model,
                         int64_t n_heads,
                         int64_t max_context_len,
                         int64_t ff_hidden_dim,
                         double dropout = 0.0)
        : ln1_(register_module(
        "ln1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{d_model})))),
        ln2_(register_module(
            "ln2",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{d_model}))
        )),
        attention_(register_module(
            "attention",
            MultiHeadCausalSelfAttention(d_model, n_heads, max_context_len)
        )),
        feedforward_(register_module(
            "feedforward",
            FeedForward(d_model, ff_hidden_dim, dropout)
        )) {}

    torch::Tensor forward(const torch::Tensor& x) {
        // x: [B, T, d_model]

        auto h = x + attention_->forward(ln1_->forward(x));
        h = h + feedforward_->forward(ln2_->forward(h));

        return h;
    }

private:
    torch::nn::LayerNorm ln1_{nullptr};
    torch::nn::LayerNorm ln2_{nullptr};

    MultiHeadCausalSelfAttention attention_{nullptr};
    FeedForward feedforward_{nullptr};
};

TORCH_MODULE(TransformerBlock);


} // namespace models
