#include <iostream>
#include <algorithm>
#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/transforms/stack.h>

#include "data/corpus.h"
#include "data/char_tokenizer.h"
#include "data/sequence_dataset.h"

int main() {
    const std::string train_path = "data/raw/TinyStoriesV2-GPT4-train.txt";
    const std::string val_path = "data/raw/TinyStoriesV2-GPT4-valid.txt";

    const size_t max_train_chars = 5'000'000; // 0 = no limit
    const size_t max_val_chars = 500'000;     // 0 = no limit

    const int64_t context_len = 128;
    const int64_t batch_size = 32;

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

    data::SequenceDataset train_dataset(train_ids, context_len);
    data::SequenceDataset val_dataset(val_ids, context_len);

    auto train_loader = torch::data::make_data_loader(
        train_dataset.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );

    auto val_loader = torch::data::make_data_loader(
        val_dataset.map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );

    std::cout << "Train chars: " << corpus.train_text.size() << "\n";
    std::cout << "Valid chars: " << corpus.val_text.size() << "\n";
    std::cout << "Vocab size:  " << tokenizer.vocab_size() << "\n";
    std::cout << "Train ids:   " << train_ids.size() << "\n";
    std::cout << "Valid ids:   " << val_ids.size() << "\n";

    auto train_it = train_loader->begin();
    if (train_it != train_loader->end()) {
        auto batch = *train_it;
        std::cout << "Train batch x shape: " << batch.data.sizes() << "\n";
        std::cout << "Train batch y shape: " << batch.target.sizes() << "\n";
    }

    auto valid_it = val_loader->begin();
    if (valid_it != val_loader->end()) {
        auto batch = *valid_it;
        std::cout << "Valid batch x shape: " << batch.data.sizes() << "\n";
        std::cout << "Valid batch y shape: " << batch.target.sizes() << "\n";
    }

    std::vector<int64_t> sample_ids(
        train_ids.begin(),
        train_ids.begin() + std::min<size_t>(200, train_ids.size())
    );

    std::cout << "\nDecoded sample:\n";
    std::cout << tokenizer.decode(sample_ids) << "\n";

    return 0;
}
