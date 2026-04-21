#pragma once

#include <cstdint>
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <torch/utils.h>
#include "../models/bigram_lm.h"
#include "batching.h"

namespace train {

struct BigramTrainConfig {
    int64_t batch_size = 64;
    int64_t context_len = 128;
    int64_t max_steps = 3000;
    int64_t eval_interval = 200;
    int64_t eval_iters = 50;
    double learning_rate = 1e-3;
    double weight_decay = 1e-2;
    std::string checkpoint_path = "checkpoints/bigram.pt";
};

struct LossEstimate {
    double train_loss = 0.0;
    double val_loss = 0.0;
};

class BigramTrainer {
public:
    BigramTrainer(models::BigramLM model,
                  BigramTrainConfig config,
                  torch::Device device)
        : model_(std::move(model)),
          config_(std::move(config)),
          device_(device),
          optimizer_(model_->parameters(),
               torch::optim::AdamWOptions(config_.learning_rate).weight_decay(config_.weight_decay)) {
        model_->to(device_);
    }

    LossEstimate estimate_loss(const torch::Tensor& train_tokens,
                               const torch::Tensor& val_tokens) {
        torch::NoGradGuard no_grad;
        model_->eval();

        auto estimate_on = [&](const torch::Tensor& tokens) {
            double acc = 0.0;
            for (int64_t i = 0; i < config_.eval_iters; i++) {
                auto batch = get_batch(tokens, config_.batch_size, config_.context_len, device_);
                auto logits = model_->forward(batch.x);
                auto loss = model_->loss(logits, batch.y);
                acc += loss.item<double>();
            }
            return acc / static_cast<double>(config_.eval_iters);
        };

        LossEstimate out;
        out.train_loss = estimate_on(train_tokens);
        out.val_loss = estimate_on(val_tokens);

        model_->train();
        return out;
    }

    void train_loop(const torch::Tensor& train_tokens,
                    const torch::Tensor& val_tokens) {
        model_->train();

        for (int64_t step = 0; step < config_.max_steps; step++) {
            if (step % config_.eval_interval == 0 || step == config_.max_steps - 1) {
                auto losses = estimate_loss(train_tokens, val_tokens);
                std::cout
                    << "[step " << step << "] "
                    << "train loss = " << losses.train_loss
                    << ", valid loss = " << losses.val_loss
                    << "\n";
            }

            auto batch = get_batch(train_tokens, config_.batch_size, config_.context_len, device_);

            optimizer_.zero_grad();
            auto logits = model_->forward(batch.x);
            auto loss = model_->loss(logits, batch.y);
            loss.backward();
            optimizer_.step();
        }
    }

    void save_checkpoint() {
        torch::save(model_, config_.checkpoint_path);
    }

    void load_checlpoint() {
        torch::load(model_, config_.checkpoint_path);
        model_->to(device_);
    }

    models::BigramLM model() const {
        return model_;
    }

private:
    models::BigramLM model_;
    BigramTrainConfig config_;
    torch::Device device_;
    torch::optim::AdamW optimizer_;
};

} // namespace train
