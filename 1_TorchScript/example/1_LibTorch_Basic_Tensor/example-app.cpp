#include <torch/torch.h>
#include <iostream>

torch::Tensor calculate_with_libtorch(const torch::Tensor &matrix, const torch::Tensor &vector)
{
    std::cout << matrix << vector << std::endl;
    return torch::matmul(matrix, vector);
}

int main()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    torch::Tensor matrix = torch::rand({3, 3}, torch::kFloat32);
    torch::Tensor vector = torch::rand({3}, torch::kFloat32);
    std::cout << matrix << std::endl;
    std::cout << vector << std::endl;

    torch::Tensor result_libtorch;
    result_libtorch = torch::matmul(matrix, vector);
    std::cout << result_libtorch << std::endl;

    torch::Tensor result_libtorch2;
    result_libtorch2 = calculate_with_libtorch(matrix, vector);
    std::cout << result_libtorch2 << std::endl;

    if (torch::allclose(result_libtorch, result_libtorch2))
    {
        std::cout << "The results are equal." << std::endl;
    }
    else
    {
        std::cout << "The results are not equal." << std::endl;
    }

    return 0;
}