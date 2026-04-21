#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <iostream>
#include <torch/torch.h>
#include "models/embeddings.h"

int main() {
    torch::manual_seed(42);

    const int64_t vocab_size = 59;
    const int64_t d_model = 128;
    const int64_t max_context_len = 128;

    models::InputEmbeddings emb(vocab_size, d_model, max_context_len);

    // mock token batch: [B, T]
    auto idx = torch::tensor(
        {{1, 2, 3, 4, 5},
                                            {5, 4, 3, 2, 1}},
        torch::TensorOptions().dtype(torch::kLong));

    auto x = emb->forward(idx);

    std::cout << "idx shape: " << idx.sizes() << "\n";
    std::cout << "x shape: " << x.sizes() << "\n"; // expected: [2, 5, 128]

    return 0;
}
