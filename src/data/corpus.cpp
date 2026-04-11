#include "corpus.h"
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace data {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string normalize_text(const std::string& text) {
    std::string out;
    out.reserve(text.size());

    bool prev_space = false;

    for (unsigned char ch : text) {
        char c = static_cast<char>(ch);

        if (c == '\r') continue; // CRLF -> LF
        if (c == '\t') c = ' '; // tab -> space

        if (c == ' ') {
            if (prev_space) continue;
            prev_space = true;
            out.push_back(c);
            continue;
        }

        if (c == '\n') {
            prev_space = false;
            out.push_back(c);
            continue;
        }

        // convert ASCII to lowercase
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(std::tolower(c));
        }

        prev_space = false;
        out.push_back(c);
    }

    return out;
}

static std::string maybe_truncate(std::string text, size_t max_chars) {
    if (max_chars > 0 && text.size() > max_chars) {
        text.resize(max_chars);
    }
    return text;
}

CorpusFiles load_corpus_files(
    const std::string &train_path,
    const std::string &val_path,
    size_t max_train_chars,
    size_t max_val_chars
) {
    CorpusFiles corpus;
    corpus.train_text = maybe_truncate(normalize_text(read_file(train_path)), max_train_chars);
    corpus.val_text = maybe_truncate(normalize_text(read_file(val_path)), max_val_chars);

    if (corpus.train_text.empty()) {
        throw std::runtime_error("Train text is empty after normalization.");
    }
    if (corpus.val_text.empty()) {
        throw std::runtime_error("Validation text is emtpy after normalization.");
    }

    return corpus;
}

} // namespace data
