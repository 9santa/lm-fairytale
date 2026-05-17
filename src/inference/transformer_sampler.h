#pragma once

#include <torch/torch.h>
#include <tuple>

#include "models/transformer_lm.h"

namespace inference {

using torch::indexing::Slice;

inline torch::Tensor generate_transformer(models::TransformerLM model,
                                          torch::Tensor idx,
                                          int64_t max_new_tokens,
                                          int64_t max_context_len,
                                          double temperature = 1.0,
                                          int64_t top_k = -1) {
    torch::NoGradGuard no_grad;
    model->eval();

    for (int64_t step = 0; step < max_new_tokens; step++) {
        torch::Tensor idx_cond = idx;

        // Transformer has a fixed max context because of positional embeddings
        // and the causal mask.
        if (idx.size(1) > max_context_len) {
            idx_cond = idx.index({
                Slice(),
                Slice(idx.size(1) - max_context_len, idx.size(1))
            });
        }

        // logits: [B, T, vocab_size]
        auto logits = model->forward(idx_cond);

        // last_logits: [B, vocab_size]
        auto last_logits = logits.index({Slice(), -1, Slice()});

        if (temperature <= 0.0) {
            auto next_token = std::get<1>(
                last_logits.max(/*dim=*/-1, /*keepdim=*/true)
            );

            idx = torch::cat({idx, next_token}, /*dim=*/1);
            continue;
        }

        last_logits = last_logits / temperature;

        if (top_k > 0 && top_k < last_logits.size(-1)) {
            auto topk = torch::topk(last_logits, /*k=*/top_k, /*dim=*/-1);

            auto kth_values = std::get<0>(topk)
                .index({Slice(), top_k - 1})
                .unsqueeze(-1);

            last_logits = torch::where(
                last_logits < kth_values,
                torch::full_like(last_logits, -1e9),
               last_logits
            );
        }

        auto probs = torch::softmax(last_logits, /*dim=*/-1);

        auto next_token = torch::multinomial(probs, /*num_samples=*/1);

        idx = torch::cat({idx, next_token}, /*dim=*/1);
    }

    return idx;
}

} // namespace inference
