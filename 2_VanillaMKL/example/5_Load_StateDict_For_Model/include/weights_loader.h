#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp> // Include the JSON library header

using json = nlohmann::json;

json load_weights(const std::string &file_path);

// Function to load a vector from JSON
std::vector<float> load_vector(const json &j);

// Function to load a matrix from JSON
std::vector<std::vector<float>> load_matrix(const json &j);

// Function to load a flattened matrix from JSON
std::vector<float> load_matrix_to_vector(const json &j);

// Function to debug and print std::vector<float>
void debug_vector(const std::vector<float> &vec, const std::string &name = "");

// Function to debug and print std::vector<std::vector<float>>
void debug_matrix(const std::vector<std::vector<float>> &mat, const std::string &name = "");

// Function to debug and print std::vector<float> as std::vector<std::vector<float>>
void debug_flatten_matrix(const std::vector<float> &flattened_matrix, int num_cols, const std::string &name = "");

#endif // WEIGHTS_LOADER_H
