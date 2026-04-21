#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace models {

namespace F = torch::nn::functional;

// BigramLMImpl name is for compatability with Torch API
struct BigramLMImpl : torch::nn::Module {
    explicit BigramLMImpl(int64_t vocab_size)
        : vocab_size_(vocab_size),
          token_embedding_table_(register_module(
            "token_embedding_table",
            torch::nn::Embedding(
                torch::nn::EmbeddingOptions(vocab_size, vocab_size)))) {}

    torch::Tensor forward(const torch::Tensor& idx) {
        // idx: [B, T]; B=batch size, T=context length, V=vocab size
        // logits: [B, T, V]
        return token_embedding_table_->forward(idx);
    }

    torch::Tensor loss(const torch::Tensor& logits,
                       const torch::Tensor& targets) const {
        // logits: [B, T, V]
        // targets [B, T]
        const auto B = logits.size(0);
        const auto T = logits.size(1);

        auto logits_flat = logits.reshape({B*T, vocab_size_});
        auto targets_falt = targets.reshape({B*T});

        return F::cross_entropy(logits_flat, targets_falt);
    }

    int64_t vocab_size() const {
        return vocab_size_;
    }

private:
    int64_t vocab_size_;
    torch::nn::Embedding token_embedding_table_{nullptr};
};

TORCH_MODULE(BigramLM);

} // namespace models
