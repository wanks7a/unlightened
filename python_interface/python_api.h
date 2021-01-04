#ifndef _PYTHON_API_H_
#define _PYTHON_API_H_

extern "C"
{
    __declspec(dllexport) bool predict(const char*, const float*, size_t, float*, size_t);
    __declspec(dllexport) void print_model(const char*);
    __declspec(dllexport) void print_input_shape(const char*);
    __declspec(dllexport) void print_output_shape(const char*);
    __declspec(dllexport) void get_input_shape(const char*);
    __declspec(dllexport) void get_output_shape(const char*);
}

#endif // !_PYTHON_API_H_
