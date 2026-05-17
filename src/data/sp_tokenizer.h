#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <sentencepiece_processor.h>

namespace data {

class SentencePieceTokenizer {
public:
    SentencePieceTokenizer() = default;

    // Train a SentencePiece model from a text file.
    static void train_model(
        const std::string& input_path,
        const std::string& model_prefix,
        int vocab_size,
        const std::string& model_type = "bpe"
    );

    // Load an existing .model file.
    void load(const std::string& model_path);

    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int64_t>& ids) const;

    int64_t vocab_size() const;
    bool loaded() const { return loaded_; }

private:
    sentencepiece::SentencePieceProcessor processor_;
    bool loaded_ = false;
};


} // namespace data
