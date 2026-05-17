#include "sp_tokenizer.h"

#include <stdexcept>
#include <sstream>

#include <sentencepiece_trainer.h>

namespace data {

void SentencePieceTokenizer::train_model(
    const std::string &input_path,
    const std::string &model_prefix,
    int vocab_size,
    const std::string& model_type
) {
    std::ostringstream args;

    /*No special BOS/EOS handling for now*/
    args
        << "--input=" << input_path
        << " --model_prefix=" << model_prefix
        << " --vocab_size=" << vocab_size
        << " --model_type=" << model_type
        << " --character_coverage=1.0"
        << " --bos_id=-1"
        << " --eos_id=-1"
        << " --pad_id=-1"
        << " --unk_id=0"
        << " --hard_vocab_limit=false";

    auto status = sentencepiece::SentencePieceTrainer::Train(args.str());

    if (!status.ok()) {
        throw std::runtime_error(
            "SentencePiece training failed: " + status.ToString()
        );
    }
}

void SentencePieceTokenizer::load(const std::string& model_path) {
    auto status = processor_.Load(model_path);

    if (!status.ok()) {
        throw std::runtime_error(
            "Failed to load SentencePiece model: " + status.ToString()
        );
    }

    loaded_ = true;
}

std::vector<int64_t> SentencePieceTokenizer::encode(const std::string& text) const {
    if (!loaded_) {
        throw std::runtime_error("SentencePieceTokenizer is not loaded");
    }

    std::vector<int> ids32;
    auto status = processor_.Encode(text, &ids32);

    if (!status.ok()) {
        throw std::runtime_error(
            "SentencePiece encode failed: " + status.ToString()
        );
    }

    // Convert to int64 for Libtorch.
    std::vector<int64_t> ids;
    ids.reserve(ids32.size());

    for (int id : ids32) ids.push_back(static_cast<int64_t>(id));

    return ids;
}

std::string SentencePieceTokenizer::decode(const std::vector<int64_t>& ids) const {
    if (!loaded_) {
        throw std::runtime_error("SentencePieceTokenizer is not loaded");
    }

    std::vector<int> ids32;
    ids32.reserve(ids.size());

    for (int64_t id : ids) ids32.push_back(static_cast<int>(id));

    std::string text;
    auto status = processor_.Decode(ids32, &text);

    if (!status.ok()) {
        throw std::runtime_error(
            "SentencePiece decode failed: " + status.ToString()
        );
    }

    return text;
}

int64_t SentencePieceTokenizer::vocab_size() const {
    if (!loaded_) {
        throw std::runtime_error("SentencePieceTokenizer is not loaded");
    }

    return static_cast<int64_t>(processor_.GetPieceSize());
}


} // namespace data
