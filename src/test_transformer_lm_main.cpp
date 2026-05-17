#include <iostream>
#include <torch/torch.h>

#include "models/transformer_lm.h"

int main() {
    torch::manual_seed(42);

    models::TransformerLMConfig config;
    config.vocab_size = 59;
    config.max_context_len = 128;
    config.d_model = 128;
    config.n_heads = 4;
    config.n_layers = 2;
    config.ff_hidden_dim = 4 * config.d_model;
    config.dropout = 0.0;

    models::TransformerLM model(config);

    auto idx = torch::tensor(
        {{1, 2, 3, 4, 5},
         {5, 4, 3, 2, 1}},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto targets = torch::tensor(
        {{2, 3, 4, 5, 6},
         {4, 3, 2, 1, 0}},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto logits = model->forward(idx);
    auto loss = model->loss(logits, targets);

    std::cout << "idx shape:     " << idx.sizes() << "\n";
    std::cout << "logits shape:  " << logits.sizes() << "\n";
    std::cout << "targets shape: " << targets.sizes() << "\n";
    std::cout << "loss:          " << loss.item<double>() << "\n";

    return 0;
}
