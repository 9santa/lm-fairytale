#include "sequence_dataset.h"
#include <stdexcept>

namespace data {

torch::data::Example<> SequenceDataset::get(size_t index) {
    size_t start = index;
    size_t end = start + context_len_ + 1;

    if (end > tokens_.size()) {
        throw std::runtime_error("SequenceDataset index out of range.");
    }

    // Get an example at index
    std::vector<int64_t> x(tokens_.begin() + start, tokens_.begin() + end - 1);
    std::vector<int64_t> y(tokens_.begin() + start + 1, tokens_.begin() + end);

    // Make tensors from the example
    auto x_tensor = torch::tensor(x, torch::kInt64);
    auto y_tensor = torch::tensor(y, torch::kInt64);

    return {x_tensor, y_tensor};
}

torch::optional<size_t> SequenceDataset::size() const {
    if (tokens_.size() <= static_cast<size_t>(context_len_)) {
        return 0;
    }
    return tokens_.size() - context_len_;
}

} // namespace data
