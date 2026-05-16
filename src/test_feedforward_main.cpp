#include <iostream>
#include <torch/torch.h>

#include "models/embeddings.h"
#include "models/feedforward.h"

int main() {
    torch::manual_seed(42);

    const int64_t vocab_size = 59;
    const int64_t d_model = 128;
    const int64_t hidden_dim = 4 * d_model;
    const int64_t max_context_len = 128;

    models::InputEmbeddings embeddings(vocab_size, d_model, max_context_len);
    models::FeedForward ff(d_model, hidden_dim, 0.0);

    auto idx = torch::tensor(
        {{1, 2, 3, 4, 5},
         {5, 4, 3, 2, 1}},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto x = embeddings->forward(idx); // [2, 5, 128]
    auto y = ff->forward(x);           // [2, 5, 128]

    std::cout << "idx shape: " << idx.sizes() << "\n";
    std::cout << "x shape:   " << x.sizes() << "\n";
    std::cout << "y shape:   " << y.sizes() << "\n";

    return 0;
}
