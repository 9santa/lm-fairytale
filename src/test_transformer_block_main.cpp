#include <iostream>
#include <torch/torch.h>

#include "models/embeddings.h"
#include "models/transformer_block.h"

int main() {
    torch::manual_seed(42);

    const int64_t vocab_size = 59;
    const int64_t d_model = 128;
    const int64_t n_heads = 4;
    const int64_t max_context_len = 128;
    const int64_t ff_hidden_dim = 4 * d_model;

    models::InputEmbeddings embeddings(vocab_size, d_model, max_context_len);

    models::TransformerBlock block(
        d_model,
        n_heads,
        max_context_len,
        ff_hidden_dim,
        0.0
    );

    auto idx = torch::tensor(
        {{1, 2, 3, 4, 5},
         {5, 4, 3, 2, 1}},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto x = embeddings->forward(idx); // [2, 5, 128]
    auto y = block->forward(x);        // [2, 5, 128]

    std::cout << "idx shape: " << idx.sizes() << "\n";
    std::cout << "x shape:   " << x.sizes() << "\n";
    std::cout << "y shape:   " << y.sizes() << "\n";

    return 0;
}
