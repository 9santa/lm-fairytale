cmake_minimum_required(VERSION 3.18)
project(lm_fairytale)

find_package(Torch REQUIRED)

# add_executable(lm_fairytale
#     src/data_prep_main.cpp
#     src/data/char_tokenizer.cpp
#     src/data/corpus.cpp
#     src/data/sequence_dataset.cpp
# )
# target_link_libraries(lm_fairytale "${TORCH_LIBRARIES}")
# set_property(TARGET lm_fairytale PROPERTY CXX_STANDARD 20)

# add_executable(train_bigram
#     src/train_bigram_main.cpp
#     src/data/char_tokenizer.cpp
#     src/data/corpus.cpp
# )
# target_include_directories(train_bigram PRIVATE src)
# target_link_libraries(train_bigram "${TORCH_LIBRARIES}")
# set_property(TARGET train_bigram PROPERTY CXX_STANDARD 20)


# Test: Embeddings
add_executable(test_embeddings
    src/test_embeddings_main.cpp
)
target_include_directories(test_embeddings PRIVATE src)
target_link_libraries(test_embeddings "${TORCH_LIBRARIES}")
set_property(TARGET test_embeddings PROPERTY CXX_STANDARD 20)


# Test: SingleHeadAttention
add_executable(test_attention
    src/test_attention_main.cpp
)

target_include_directories(test_attention PRIVATE src)
target_link_libraries(test_attention "${TORCH_LIBRARIES}")
set_property(TARGET test_attention PROPERTY CXX_STANDARD 20)


# Test: MultiHeadAttention
add_executable(test_multihead_attention
    src/test_multihead_attention_main.cpp
)

target_include_directories(test_multihead_attention PRIVATE src)
target_link_libraries(test_multihead_attention "${TORCH_LIBRARIES}")
set_property(TARGET test_multihead_attention PROPERTY CXX_STANDARD 20)


# Test: FeedForward net
add_executable(test_feedforward
    src/test_feedforward_main.cpp
)

target_include_directories(test_feedforward PRIVATE src)
target_link_libraries(test_feedforward "${TORCH_LIBRARIES}")
set_property(TARGET test_feedforward PROPERTY CXX_STANDARD 20)


# Test: TransformerBlock
add_executable(test_transformer_block
    src/test_transformer_block_main.cpp
)

target_include_directories(test_transformer_block PRIVATE src)
target_link_libraries(test_transformer_block "${TORCH_LIBRARIES}")
set_property(TARGET test_transformer_block PROPERTY CXX_STANDARD 20)
