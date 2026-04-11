#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>


namespace data {

class CharTokenizer {
public:
    void fit(const std::string& text);

    std::vector<int64_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int64_t>& ids) const;

    int64_t vocab_size() const;
    bool fitted() const { return fitted_; }

private:
    std::vector<char> id_to_char_;
    std::unordered_map<char, int64_t> char_to_id_;
    bool fitted_ = false;
};

} // namespace data
