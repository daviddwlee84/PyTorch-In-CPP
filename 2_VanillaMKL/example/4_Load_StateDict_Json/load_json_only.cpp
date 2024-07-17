#include <iostream>
#include "weights_loader.h"

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: load_json <path-to-state_dict_json>\n";
        return -1;
    }

    json weights;
    try
    {
        // Load weights from JSON file
        std::string file_path = argv[1];
        weights = load_weights(file_path);
    }
    catch (...)
    {
        // https://stackoverflow.com/questions/315948/c-catching-all-exceptions
        std::cerr << "error loading the json\n";
        std::exception_ptr p = std::current_exception();
        std::clog << (p ? p.__cxa_exception_type()->name() : "null") << std::endl;
        return -1;
    }

    std::cout << weights << std::endl;
}
