#include <iostream>
#include <torch/torch.h>

#include "models/embeddings.h"
#include "models/self_attention.h"

int main() {
    torch::manual_seed(42);

    using i64 = int64_t;
    const i64 vocab_size = 59;
    const i64 d_model = 128;
    const i64 head_dim = 32;
    const i64 max_context_len = 128;

    models::InputEmbeddings embeddings(vocab_size, d_model, max_context_len);
    models::CausalSelfAttentionHead attention(d_model, head_dim, max_context_len);

    auto idx = torch::tensor(
    {{1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1}},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto x = embeddings->forward(idx);
    auto y = attention->forward(x);

    std::cout << "idx shape: " << idx.sizes() << "\n";
    std::cout << "x shape:   " << x.sizes() << "\n";
    std::cout << "y shape:   " << y.sizes() << "\n";

    return 0;
}
