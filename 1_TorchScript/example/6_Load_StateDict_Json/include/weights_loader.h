#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp> // Include the JSON library header

using json = nlohmann::json;

json load_weights(const std::string &file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    json weights;
    file >> weights;
    return weights;
}

#endif // WEIGHTS_LOADER_H
