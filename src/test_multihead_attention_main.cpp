#include <iostream>
#include <torch/torch.h>

#include "models/embeddings.h"
#include "models/self_attention.h"

int main() {
    torch::manual_seed(42);

    const int64_t vocab_size = 59;
    const int64_t d_model = 128;
    const int64_t n_heads = 4;
    const int64_t max_context_len = 128;

    models::InputEmbeddings embeddings(vocab_size, d_model, max_context_len);
    models::MultiHeadCausalSelfAttention attention(
        d_model,
        n_heads,
        max_context_len
    );

    auto idx = torch::tensor(
        {{1, 2, 3, 4, 5},
    {5, 4, 3, 2, 1}},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto x = embeddings->forward(idx);
    auto y = attention->forward(x);

    std::cout << "idx shape: " << idx.sizes() << "\n";
    std::cout << "x shape: " << x.sizes() << "\n";
    std::cout << "y shape: " << y.sizes() << "\n";

    return 0;
}
