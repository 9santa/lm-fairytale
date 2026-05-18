#include <filesystem>

#include <torch/torch.h>

#include "data/corpus.h"
#include "data/sp_tokenizer.h"
#include "train/batching.h"

int main() {
    try {
        const std::string train_path =
            "data/raw/TinyStoriesV2-GPT4-train.txt";

        const std::string valid_path =
            "data/raw/TinyStoriesV2-GPT4-valid.txt";

        const size_t max_train_chars = 5'000'000;
        const size_t max_valid_chars = 500'000;

        const std::string spm_train_text_path =
            "data/spm/tinystories_train_5m.txt";

        const std::string spm_model_prefix =
            "data/spm/tinystories_bpe_1024";

        const std::string spm_model_path =
            spm_model_prefix + ".model";

        const int vocab_size = 1024;

        auto corpus = data::load_corpus_files(
            train_path,
            valid_path,
            max_train_chars,
            max_valid_chars
        );

        std::cout << "Train chars: " << corpus.train_text.size() << "\n";
        std::cout << "Valid chars: " << corpus.val_text.size() << "\n";

        data::write_file(spm_train_text_path, corpus.train_text);

        if (!std::filesystem::exists(spm_model_path)) {
            std::cout << "Training SentencePiece model...\n";

            data::SentencePieceTokenizer::train_model(
                spm_train_text_path,
                spm_model_prefix,
                vocab_size,
                "bpe"
            );
        } else {
            std::cout << "SentencePiece model already exists, loading it.\n";
        }

        data::SentencePieceTokenizer tokenizer;
        tokenizer.load(spm_model_path);

        std::cout << "SP vocab size: " << tokenizer.vocab_size() << "\n";

        auto train_ids = tokenizer.encode(corpus.train_text);
        auto valid_ids = tokenizer.encode(corpus.val_text);

        std::cout << "Train token ids: " << train_ids.size() << "\n";
        std::cout << "Valid token ids: " << valid_ids.size() << "\n";

        const double train_chars_per_token =
            static_cast<double>(corpus.train_text.size()) /
            static_cast<double>(train_ids.size());

        const double valid_chars_per_token =
            static_cast<double>(corpus.val_text.size()) /
            static_cast<double>(valid_ids.size());

        std::cout << "Train chars/token: " << train_chars_per_token << "\n";
        std::cout << "Valid chars/token: " << valid_chars_per_token << "\n";

        std::vector<int64_t> sample_ids(
            train_ids.begin(),
            train_ids.begin() + std::min<size_t>(64, train_ids.size())
        );

        std::cout << "\nDecoded sample:\n";
        std::cout << tokenizer.decode(sample_ids) << "\n";

        auto train_tokens = torch::tensor(
            train_ids,
            torch::TensorOptions().dtype(torch::kLong)
        );

        const int64_t batch_size = 4;
        const int64_t context_len = 32;

        auto batch = train::get_batch(
            train_tokens,
            batch_size,
            context_len,
            torch::kCPU
        );

        std::cout << "\nBatch x shape: " << batch.x.sizes() << "\n";
        std::cout << "Batch y shape: " << batch.y.sizes() << "\n";

        return 0;
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
