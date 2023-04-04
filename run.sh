#!/bin/bash

# Run main file
python main.py \
--seed_value 11 \
--create_input_data False \
--dataset_name "covid-19_tweets" \
--dataset_dir "datasets/covid-19_tweets/" \
--model_name "brnn" \
--train True \
--train_epochs 10 \
--batch_size 50 \
--word_embeddings "word2vec" \
--pretrained_word_embeddings_path "datasets/pre_trained_word_vectors/word2vec/GoogleNews-vectors-negative300.bin" \
--fine_tune_word_embeddings False \
--create_embedding_mask True \
--embedding_dim 300 \
--sequence_layer_units 128 \
--num_of_bayesian_samples 10
# --no_of_filters 100 \
# --filter_sizes [3,4,5] \