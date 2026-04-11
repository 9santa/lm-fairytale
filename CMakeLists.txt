cmake_minimum_required(VERSION 3.18)
project(lm_fairy_tails)

find_package(Torch REQUIRED)

add_executable(lm_fairy_tails
    src/data_prep_main.cpp
    src/data/char_tokenizer.cpp
    src/data/corpus.cpp
    src/data/sequence_dataset.cpp
)
target_link_libraries(lm_fairy_tails "${TORCH_LIBRARIES}")
set_property(TARGET lm_fairy_tails PROPERTY CXX_STANDARD 20)
