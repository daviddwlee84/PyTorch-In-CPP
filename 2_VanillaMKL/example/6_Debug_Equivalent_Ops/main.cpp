#include <torch/torch.h>
#include <mkl.h>
#include <iostream>
#include <vector>

void tensor_to_mkl_vector(const torch::Tensor &tensor, float *&vector, int &length)
{
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    length = tensor.numel();
    vector = tensor.data_ptr<float>();
}

void tensor_to_mkl_vector(const torch::Tensor &tensor, std::vector<float> &vector)
{
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    int length = tensor.numel();
    vector.resize(length);
    std::memcpy(vector.data(), tensor.data_ptr<float>(), length * sizeof(float));
}

void tensor_to_mkl_matrix(const torch::Tensor &tensor, float *&matrix, int &rows, int &cols)
{
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    rows = tensor.size(0);
    cols = tensor.size(1);
    matrix = tensor.data_ptr<float>();
}

void tensor_to_mkl_matrix(const torch::Tensor &tensor, std::vector<float> &matrix, int &rows, int &cols)
{
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    rows = tensor.size(0);
    cols = tensor.size(1);
    matrix.resize(rows * cols);
    std::memcpy(matrix.data(), tensor.data_ptr<float>(), rows * cols * sizeof(float));
}

void calculate_with_mkl(const torch::Tensor &matrix, const torch::Tensor &vector, torch::Tensor &result)
{
    float *mkl_matrix, *mkl_vector, *mkl_result;
    int rows, cols, length;

    tensor_to_mkl_matrix(matrix, mkl_matrix, rows, cols);
    tensor_to_mkl_vector(vector, mkl_vector, length);

    // Perform matrix-vector multiplication: result = matrix * vector
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0, mkl_matrix, cols, mkl_vector, 1, 0.0, mkl_result, 1);

    tensor_to_mkl_vector(result, mkl_result, length);
}

// Function to debug and print std::vector<float>
void debug_vector(const std::vector<float> &vec, const std::string &name = "")
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
void debug_matrix(const std::vector<std::vector<float>> &mat, const std::string &name = "")
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

void debug_flatten_matrix(const std::vector<float> &flattened_matrix, int num_cols, const std::string &name = "")
{
    if (!name.empty())
    {
        std::cout << name << ":\n";
    }

    int num_rows = flattened_matrix.size() / num_cols;

    std::cout << "[\n";
    for (int i = 0; i < num_rows; ++i)
    {
        std::cout << "  [";
        for (int j = 0; j < num_cols; ++j)
        {
            // std::cout << std::setw(10) << flattened_matrix[i * num_cols + j] << " ";
            std::cout << flattened_matrix[i * num_cols + j];
            if (j < num_cols - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i < num_rows - 1)
        {
            std::cout << ",\n";
        }
    }
    std::cout << "\n]" << std::endl;
    std::cout << "(size: " << flattened_matrix.size() << ")" << std::endl;
    std::cout << "(convert size: " << num_rows << ", " << num_cols << ")" << std::endl;
}

torch::Tensor calculate_with_libtorch(const torch::Tensor &matrix, const torch::Tensor &vector)
{
    return torch::matmul(matrix, vector);
}

int main()
{
    // Initialize tensors
    torch::Tensor matrix = torch::rand({3, 3}, torch::kFloat32);
    torch::Tensor vector = torch::rand({3}, torch::kFloat32);
    torch::Tensor result_mkl = torch::zeros({3}, torch::kFloat32);
    torch::Tensor result_libtorch;

    // Print tensors for reference
    std::cout << "Matrix:\n"
              << matrix << std::endl;
    std::cout << "Vector:\n"
              << vector << std::endl;

    // float *mkl_matrix, *mkl_vector;
    std::vector<float> mkl_flatten_matrix, mkl_vector;
    int cols, rows, lengths;
    tensor_to_mkl_matrix(matrix, mkl_flatten_matrix, rows, cols);
    tensor_to_mkl_vector(vector, mkl_vector);
    debug_flatten_matrix(mkl_flatten_matrix, 3);
    debug_vector(mkl_vector);

    // Perform calculations with oneMKL
    /*
    calculate_with_mkl(matrix, vector, result_mkl);

    // Perform calculations with LibTorch
    result_libtorch = calculate_with_libtorch(matrix, vector);

    // Print results
    std::cout << "Result (oneMKL):\n"
              << result_mkl << std::endl;
    std::cout << "Result (LibTorch):\n"
              << result_libtorch << std::endl;

    // Compare results
    if (torch::allclose(result_mkl, result_libtorch))
    {
        std::cout << "The results are equal." << std::endl;
    }
    else
    {
        std::cout << "The results are not equal." << std::endl;
    }
    */

    return 0;
}
