#pragma once
#include <cstdint>
#include <torch/torch.h>
#include <vector>

namespace data {

class SequenceDataset : public torch::data::datasets::Dataset<SequenceDataset> {
public:
    SequenceDataset(std::vector<int64_t> tokens, int64_t context_len)
        : tokens_(std::move(tokens)), context_len_(context_len) {}
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    std::vector<int64_t> tokens_;
    int64_t context_len_;
};

} // namespace data
