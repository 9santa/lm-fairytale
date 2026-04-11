#pragma once
#include <string>
#include <vector>

namespace data {

struct CorpusFiles {
    /* TinyStories dataset already has a train/val split */
    std::string train_text;
    std::string val_text;
};

std::string read_file(const std::string& path);
std::string normalize_text(const std::string& text);

CorpusFiles load_corpus_files (
    const std::string& train_path,
    const std::string& val_path,
    size_t max_train_chars = 0,
    size_t max_val_chars = 0
);

} // namespace data
