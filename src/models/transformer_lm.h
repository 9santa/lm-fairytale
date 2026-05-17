#pragma once

#include <c10/core/Device.h>
#include <cstdint>
#include <string>
#include <torch/nn/functional/loss.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/options/linear.h>
#include <torch/nn/options/normalization.h>
#include <vector>

#include <torch/torch.h>

#include "models/embeddings.h"
#include "models/transformer_block.h"

namespace models {

namespace F = torch::nn::functional;

struct TransformerLMConfig {
    int64_t vocab_size;
    int64_t max_context_len;

    int64_t d_model = 128;
    int64_t n_heads = 4;
    int64_t n_layers = 2;
    int64_t ff_hidden_dim = 4 * d_model;

    double dropout = 0.0;
};

struct TransformerLMImpl : torch::nn::Module {
    explicit TransformerLMImpl(TransformerLMConfig config)
        : config_(config),
        embeddings_(register_module(
            "embeddings",
            InputEmbeddings(config.vocab_size, config.d_model, config.max_context_len)
        )),
        final_ln_(register_module(
            "final_ln",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions(std::vector<int64_t>{config.d_model}))
        )),
        lm_head_(register_module(
            "lm_head",
            torch::nn::Linear(torch::nn::LinearOptions(config.d_model, config.vocab_size))
        )) {

        TORCH_CHECK(config_.vocab_size > 0, "vocab_size must be positive");
        TORCH_CHECK(config_.max_context_len > 0, "max_context_len must be positive");
        TORCH_CHECK(config_.d_model > 0, "d_model must be positive");
        TORCH_CHECK(config_.n_heads > 0, "n_heads must be positive");
        TORCH_CHECK(config_.n_layers > 0, "n_layers must be positive");
        TORCH_CHECK(
            config_.d_model % config_.n_heads == 0,
            "d_model must be divisible by n_heads"
        );

        for (int64_t i = 0; i < config_.n_layers; i++) {
            auto block = TransformerBlock(
                config_.d_model,
                   config_.n_heads,
                   config_.max_context_len,
                   config_.ff_hidden_dim,
                   config_.dropout
            );

            blocks_.push_back(register_module(
                "block_" + std::to_string(i), block
            ));
        }
    }

    torch::Tensor forward(const torch::Tensor& idx) {
        // idx: [B, T]
        TORCH_CHECK(idx.dim() == 2, "TransformerLM expects idx of shape [B, T]");

        const auto T = idx.size(1);

        TORCH_CHECK(T <= config_.max_context_len,
                    "Sequence length ", T,
                    " exceeds max_context_len ", config_.max_context_len
        );

        // [B, T] -> [B, T, d_model]
        auto x = embeddings_->forward(idx);

        // [B, T, d_model] -> [B, T, d_model]
        for (auto& block : blocks_) {
            x = block->forward(x);
        }

        // [B, T, d_model]
        x = final_ln_->forward(x);

        // [B, T, d_model] -> [B, T, vocab_size]
        auto logits = lm_head_->forward(x);

        return logits;
    }

    torch::Tensor loss(const torch::Tensor& logits,
                       const torch::Tensor& targets) const {
        // logits: [B, T, vocab_size]
        // targets: [B, T]

        TORCH_CHECK(logits.dim() == 3, "logits must have shape [B, T, vocab_size]");
        TORCH_CHECK(targets.dim() == 2, "targets must have shape [B, T]");

        const auto B = logits.size(0);
        const auto T = logits.size(1);
        const auto V = logits.size(2);

        TORCH_CHECK(V == config_.vocab_size,
                    "Expected vocab dim = ", config_.vocab_size, ", got ", V);

        auto logits_flat = logits.reshape({B * T, V});
        auto targets_flat = targets.reshape({B * T});

        return F::cross_entropy(logits_flat, targets_flat);
    }

    int64_t vocab_size() const { return config_.vocab_size; }

    int64_t max_context_len() const { return config_.max_context_len; }

    const TransformerLMConfig& config() const { return config_; }

private:
    TransformerLMConfig config_;

    InputEmbeddings embeddings_{nullptr};
    std::vector<TransformerBlock> blocks_;

    torch::nn::LayerNorm final_ln_{nullptr};
    torch::nn::Linear lm_head_{nullptr};
};

TORCH_MODULE(TransformerLM);


} // namespace models
