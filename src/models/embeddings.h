#pragma once

#include <torch/torch.h>

namespace models {

/*
Input embedding.
Input: idx - token IDs of shape [B, T]
Output: contextual input representation of shape [B, T, d_model]
*/
struct InputEmbeddingsImpl : torch::nn::Module {
    InputEmbeddingsImpl(int64_t vocab_size,
                       int64_t d_model,
                       int64_t max_context_len)
        : d_model_(d_model),
          max_context_len_(max_context_len),
          token_embedding_(register_module(
              "token_embedding",
              torch::nn::Embedding(
                  torch::nn::EmbeddingOptions(vocab_size, d_model)))),
          position_embedding_(register_module(
            "position_embedding",
            torch::nn::Embedding(
                  torch::nn::EmbeddingOptions(max_context_len, d_model)))) {}

    torch::Tensor forward(const torch::Tensor& idx) {
        // idx: [B, T]
        TORCH_CHECK(idx.dim() == 2, "InputEmbeddings expects idx of shape [B, T].");

        const auto B = idx.size(0);
        const auto T = idx.size(1);

        TORCH_CHECK(T <= max_context_len_,
                    "Sequence length ", T,
                    " exceeds max_context_len ", max_context_len_);

        // token embeddings: [B, T, d_model]
        auto tok_emb = token_embedding_->forward(idx);

        // positions: [T] = {0, 1, 2, ..., T-1}
        auto positions = torch::arange(
            0, T,
            torch::TensorOptions().dtype(torch::kLong).device(idx.device()));

        // position embeddings: [T, d_model]
        auto pos_emb = position_embedding_->forward(positions);

        // broadcast add: [B, T, d_model] + [T, d_model] -> [B, T, d_model]
        auto x = tok_emb + pos_emb;

        return x;
    }

    int64_t d_model() const { return d_model_; }
    int64_t max_context_len() const { return max_context_len_; }

private:
    int64_t d_model_;
    int64_t max_context_len_;

    torch::nn::Embedding token_embedding_{nullptr};
    torch::nn::Embedding position_embedding_{nullptr};
};

TORCH_MODULE(InputEmbeddings);

} // namespace models
