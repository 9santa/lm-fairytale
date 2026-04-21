#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "data/corpus.h"
#include "data/char_tokenizer.h"
#include "models/bigram_lm.h"
#include "train/bigram_trainer.h"
#include "inference/sampler.h"

int main() {
    try {
        const std::string train_path = "data/raw/TinyStoriesV2-GPT4-train.txt";
        const std::string val_path = "data/raw/TinyStoriesV2-GPT4-valid.txt";

        const size_t max_train_chars = 5'000'000; // 0 = full file
        const size_t max_val_chars = 500'000;   // 0 = full file

        auto corpus = data::load_corpus_files(
            train_path,
            val_path,
            max_train_chars,
            max_val_chars
        );

        data::CharTokenizer tokenizer;
        tokenizer.fit(corpus.train_text + corpus.val_text);

        auto train_ids = tokenizer.encode(corpus.train_text);
        auto val_ids = tokenizer.encode(corpus.val_text);

        auto train_tokens = torch::tensor(train_ids, torch::TensorOptions().dtype(torch::kLong));
        auto val_tokens = torch::tensor(val_ids, torch::TensorOptions().dtype(torch::kLong));

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

        std::cout << "Device:      " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
        std::cout << "Train chars: " << corpus.train_text.size() << "\n";
        std::cout << "val chars: " << corpus.val_text.size() << "\n";
        std::cout << "Vocab size:  " << tokenizer.vocab_size() << "\n";

        auto model = models::BigramLM(tokenizer.vocab_size());

        train::BigramTrainConfig cfg;
        cfg.batch_size = 64;
        cfg.context_len = 128;
        cfg.max_steps = 3000;
        cfg.eval_interval = 200;
        cfg.eval_iters = 50;
        cfg.learning_rate = 1e-3;
        cfg.weight_decay = 1e-2;
        cfg.checkpoint_path = "checkpoints/bigram_cuda.pt";

        std::filesystem::create_directories("checkpoints");

        train::BigramTrainer trainer(model, cfg, device);
        trainer.train_loop(train_tokens, val_tokens);
        trainer.save_checkpoint();

        // Seed prompt
        std::string prompt = "once upon a time";
        auto prompt_ids = tokenizer.encode(prompt);

        auto seed = torch::tensor(prompt_ids, torch::TensorOptions().dtype(torch::kLong))
                        .unsqueeze(0)
                        .to(device);

        auto generated = inference::generate(
            trainer.model(),
            seed,
            /*max_new_tokens=*/300,
            /*temperature=*/1.0,
            /*top_k=*/20
        );

        auto generated_cpu = generated.squeeze(0).to(torch::kCPU);
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
