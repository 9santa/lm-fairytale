#pragma once

#include <ATen/ops/cat.h>
#include <c10/core/Device.h>
#include <cmath>
#include <cstdint>
#include <torch/nn/options/linear.h>
#include <torch/torch.h>

namespace models {


struct CausalSelfAttentionHeadImpl : torch::nn::Module {
    CausalSelfAttentionHeadImpl(int64_t d_model,
                                int64_t head_dim,
                                int64_t max_context_len)
        : d_model_(d_model),
        head_dim_(head_dim),
        max_context_len_(max_context_len),
        key_(register_module(
            "key",
            torch::nn::Linear(
                torch::nn::LinearOptions(d_model, head_dim).bias(false)))),
        query_(register_module(
            "query",
            torch::nn::Linear(
                torch::nn::LinearOptions(d_model, head_dim).bias(false)))),
        value(register_module(
            "value",
            torch::nn::Linear(
                torch::nn::LinearOptions(d_model, head_dim).bias(false)))) {

        // Lower triangular causal mask: [max_context_len, max_context_len]
        auto tril = torch::tril(torch::ones(
            {max_context_len_, max_context_len_},
            torch::TensorOptions().dtype(torch::kFloat32)
        ));

        tril_ = register_buffer("tril", tril);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        // x: [B, T, d_model]
        TORCH_CHECK(x.dim() == 3, "CausalSelfAttentionHead expects input of shape [B, T, d_model]");

        const auto B = x.size(0);
        const auto T = x.size(1);
        const auto C = x.size(2);

        TORCH_CHECK(C == d_model_, "Expected last dim = ", d_model_, ", got ", C);
        TORCH_CHECK(T <= max_context_len_, "Sequence length ", T, " exceeds max_context_len", max_context_len_);

        // q, k, v: [B, T, head_dim]
        auto q = query_->forward(x);
        auto k = key_->forward(x);
        auto v = value->forward(x);

        // attention scores: [B, T, T]
        auto scores = torch::matmul(q, k.transpose(1, 2)) / std::sqrt(static_cast<double>(head_dim_));

        // causal mask: keep only j <= i
        auto mask = tril_.index({torch::indexing::Slice(0, T),
                                torch::indexing::Slice(0, T)}).to(x.device());

        scores = scores.masked_fill(mask == 0, -1e9);

        // attention weights: [B, T, T]
        auto weights = torch::softmax(scores, -1);

        // output: [B, T, head_dim]
        auto out = torch::matmul(weights, v);

        return out;
    }

private:
    int64_t d_model_;
    int64_t head_dim_;
    int64_t max_context_len_;

    torch::nn::Linear key_{nullptr};
    torch::nn::Linear query_{nullptr};
    torch::nn::Linear value{nullptr};

    torch::Tensor tril_;
};

TORCH_MODULE(CausalSelfAttentionHead);


struct MultiHeadCausalSelfAttentionImpl : torch::nn::Module {
    MultiHeadCausalSelfAttentionImpl(int64_t d_model,
                                     int64_t n_heads,
                                     int64_t max_context_len)
        : d_model_(d_model),
          n_heads_(n_heads),
          max_context_len_(max_context_len),
          projection_(nullptr) {

        TORCH_CHECK(n_heads_ > 0, "n_heads must be positive");
        TORCH_CHECK(
            d_model_ % n_heads_ == 0,
     "d_model must be divisible by n_heads")

        head_dim_ = d_model_ / n_heads_;

        for (int64_t i = 0; i < n_heads_; i++) {
            auto head = CausalSelfAttentionHead(
                d_model_,
                head_dim_,
                max_context_len_
            );

            heads_.push_back(register_module(
                "head_" + std::to_string(i),
                head
            ));
        }

        projection_ = register_module(
            "projection",
            torch::nn::Linear(
                torch::nn::LinearOptions(d_model_, d_model_).bias(false)
            )
        );
    }

    torch::Tensor forward(const torch::Tensor& x) {
        // x: [B, T, d_model]
        TORCH_CHECK(
            x.dim() == 3,
            "MultiHeadCausalSelfAttention expects input of shape [B, T, d_model]"
        );

        const auto C = x.size(2);

        TORCH_CHECK(
            C == d_model_,
            "Expected last dim = ", d_model_, ", got ", C
        );

        std::vector<torch::Tensor> head_outputs;
        head_outputs.reserve(heads_.size());

        for (auto& head : heads_) {
            // each: [B, T, head_dim]
            head_outputs.push_back(head->forward(x));
        }

        // concat along feature dimension
        // [B, T, head_dim] * n_heads -> [B, T, d_model]
        auto out = torch::cat(head_outputs, /*dim=*/2);

        // final learned mixing of all heads
        // [B, T, d_model] -> [B, T, d_model]
        out = projection_->forward(out);

        return out;
    }

    int64_t d_model() const { return d_model_; }
    int64_t n_heads() const { return n_heads_; }
    int64_t head_dim() const { return head_dim_; }

private:
    int64_t d_model_;
    int64_t n_heads_;
    int64_t head_dim_;
    int64_t max_context_len_;

    std::vector<CausalSelfAttentionHead> heads_;
    torch::nn::Linear projection_{nullptr};
};

TORCH_MODULE(MultiHeadCausalSelfAttention);


} // namespace models
