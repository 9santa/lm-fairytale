#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "data/corpus.h"
#include "data/char_tokenizer.h"

#include "models/transformer_lm.h"
#include "train/transformer_trainer.h"
#include "inference/transformer_sampler.h"

int main() {
    try {
        const std::string train_path = "data/raw/TinyStoriesV2-GPT4-train.txt";
        const std::string valid_path = "data/raw/TinyStoriesV2-GPT4-valid.txt";

        // Keep this limited while smoke-testing.
        const size_t max_train_chars = 5'000'000;
        const size_t max_valid_chars = 500'000;

        auto corpus = data::load_corpus_files(
            train_path,
            valid_path,
            max_train_chars,
            max_valid_chars
        );

        data::CharTokenizer tokenizer;
        tokenizer.fit(corpus.train_text + corpus.val_text);

        auto train_ids = tokenizer.encode(corpus.train_text);
        auto valid_ids = tokenizer.encode(corpus.val_text);

        auto train_tokens = torch::tensor(
            train_ids,
            torch::TensorOptions().dtype(torch::kLong)
        );

        auto valid_tokens = torch::tensor(
            valid_ids,
            torch::TensorOptions().dtype(torch::kLong)
        );

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

        std::cout << "Device:      " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
        std::cout << "Train chars: " << corpus.train_text.size() << "\n";
        std::cout << "Valid chars: " << corpus.val_text.size() << "\n";
        std::cout << "Vocab size:  " << tokenizer.vocab_size() << "\n";

        const int64_t context_len = 128;

        models::TransformerLMConfig model_cfg;
        model_cfg.vocab_size = tokenizer.vocab_size();
        model_cfg.max_context_len = context_len;

        model_cfg.d_model = 128;
        model_cfg.n_heads = 4;
        model_cfg.n_layers = 2;
        model_cfg.ff_hidden_dim = 4 * model_cfg.d_model;
        model_cfg.dropout = 0.1;

        auto model = models::TransformerLM(model_cfg);

        train::TransformerTrainConfig train_cfg;
        train_cfg.batch_size = 32;
        train_cfg.context_len = context_len;

        // For the first smoke test, you can reduce this to 200.
        train_cfg.max_steps = 10000;
        train_cfg.eval_interval = 500;
        train_cfg.eval_iters = 50;

        train_cfg.learning_rate = 3e-4;
        train_cfg.weight_decay = 1e-2;
        train_cfg.checkpoint_path = "checkpoints/transformer2.pt";

        std::filesystem::create_directories("checkpoints");

        train::TransformerTrainer trainer(
            model,
            train_cfg,
            device
        );

        trainer.train_loop(train_tokens, valid_tokens);
        trainer.save_checkpoint();

        std::string prompt = "once upon a time";
        auto prompt_ids = tokenizer.encode(prompt);

        auto seed = torch::tensor(
            prompt_ids,
            torch::TensorOptions().dtype(torch::kLong)
        ).unsqueeze(0).to(device);

        auto generated = inference::generate_transformer(
            trainer.model(),
            seed,
            /*max_new_tokens=*/400,
            /*max_context_len=*/context_len,
            /*temperature=*/0.6,
            /*top_k=*/10
        );

        auto generated_cpu = generated
            .squeeze(0)
            .to(torch::kCPU)
            .contiguous();

        std::vector<int64_t> generated_ids(generated_cpu.numel());

        auto* ptr = generated_cpu.data_ptr<int64_t>();
        std::copy(ptr, ptr + generated_ids.size(), generated_ids.begin());

        std::cout << "\n=== GENERATED SAMPLE ===\n";
        std::cout << tokenizer.decode(generated_ids) << "\n";

        return 0;
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
