#include "char_tokenizer.h"
#include <set>
#include <stdexcept>

namespace data {

void CharTokenizer::fit(const std::string& text) {
    std::set<char> chars(text.begin(), text.end());

    id_to_char_.assign(chars.begin(), chars.end());
    char_to_id_.clear();

    for (int64_t i = 0; i < static_cast<int64_t>(id_to_char_.size()); i++) {
        char_to_id_[id_to_char_[i]] = i;
    }

    fitted_ = true;
}

std::vector<int64_t> CharTokenizer::encode(const std::string& text) const {
    if (!fitted_) {
        throw std::runtime_error("CharTokenizer is not fitted.");
    }

    std::vector<int64_t> ids;
    ids.reserve(text.size());

    for (char c : text) {
        auto it = char_to_id_.find(c);
        if (it == char_to_id_.end()) {
            throw std::runtime_error("Unknown char found during encode.");
        }
        ids.push_back(it->second);
    }

    return ids;
}

std::string CharTokenizer::decode(const std::vector<int64_t>& ids) const {
    if (!fitted_) {
        throw std::runtime_error("CharTokenizer is not fitted.");
    }

    std::string text;
    text.reserve(ids.size());

    for (int64_t id : ids) {
        if (id < 0 || id >= static_cast<int64_t>(id_to_char_.size())) {
            throw std::runtime_error("Invalid token id during decode.");
        }
        text.push_back(id_to_char_[id]);
    }

    return text;
}

int64_t CharTokenizer::vocab_size() const {
    return static_cast<int64_t>(id_to_char_.size());
}

} // namespace data
