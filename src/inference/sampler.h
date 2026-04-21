#pragma once

#include <torch/torch.h>
#include "../models/bigram_lm.h"

namespace inference {

using torch::indexing::Slice;

inline torch::Tensor generate(models::BigramLM model,
                              torch::Tensor idx,
                              int64_t max_new_tokens,
                              double temperature = 1.0,
                              int64_t top_k = -1) {
    torch::NoGradGuard no_grad; // disable gradient calculation
    model->eval(); // put model into eval mode

    for (int64_t step = 0; step < max_new_tokens; step++) {
        // idx: [B, T_current]
        auto logits = model->forward(idx);   // [B, T_current, V]
        // scores (logits) for the next token
        auto last_logits = logits.index({Slice(), -1, Slice()}); // [B, V]

        // if temp not set, just greedily take the highest-scoring token
        if (temperature <= 0.0) {
            auto next_token = std::get<1>(last_logits.max(-1, true));
            idx = torch::cat({idx, next_token}, 1); // append that token to the right of the sequence
            continue;
        }

        // temperature scaling
        last_logits = last_logits / temperature;

        if (top_k > 0 && top_k < last_logits.size(-1)) {
            auto topk = torch::topk(last_logits, top_k, -1);
            auto kth_values = std::get<0>(topk).index({Slice(), top_k-1}).unsqueeze(-1);
            /* suppresses the rest of the tokens, which are not in top-k */
            last_logits = torch::where(
                last_logits < kth_values,
                     torch::full_like(last_logits, -1e9),
                    last_logits
            );
        }

        auto probs = torch::softmax(last_logits, -1); // [B, V]
        // choose according to the probability distribution (generalized Bernoulli for V possible events instead of only 2)
        auto next_token = torch::multinomial(probs, 1); // [B, 1]
        idx = torch::cat({idx, next_token}, 1);
    }

    return idx;
}

} // namespace inference
