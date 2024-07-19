#include "../include/weights_loader.h"

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

// Function to load a vector from JSON
std::vector<float> load_vector(const json &j)
{
    return j.get<std::vector<float>>();
}
/*
{
    try
    {
        // Check if the JSON is an array
        if (j.is_array())
        {
            return j.get<std::vector<float>>();
        }
        // If it's not an array, try to get it as a single float
        else if (j.is_number())
        {
            return {j.get<float>()};
        }
        else
        {
            throw std::runtime_error("Unsupported JSON type for vector loading");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        // Return a vector initialized with one element as a fallback
        return {0.0f};
    }
}
*/

// Function to load a matrix from JSON
std::vector<std::vector<float>> load_matrix(const json &j)
{
    return j.get<std::vector<std::vector<float>>>();
}

// Function to debug and print std::vector<float>
void debug_vector(const std::vector<float> &vec, const std::string &name)
{
    if (!name.empty())
    {
        std::cout << name << ": ";
    }
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i];
        if (i < vec.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "(size: " << vec.size() << ")" << std::endl;
}

// Function to debug and print std::vector<std::vector<float>>
void debug_matrix(const std::vector<std::vector<float>> &mat, const std::string &name)
{
    if (!name.empty())
    {
        std::cout << name << ": ";
    }
    std::cout << "[\n";
    for (size_t i = 0; i < mat.size(); ++i)
    {
        std::cout << "  [";
        for (size_t j = 0; j < mat[i].size(); ++j)
        {
            std::cout << mat[i][j];
            if (j < mat[i].size() - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < mat.size() - 1)
        {
            std::cout << ",\n";
        }
    }
    std::cout << "\n]" << std::endl;
    std::cout << "(size: " << mat.size() << ", " << mat[0].size() << ")" << std::endl;
}
