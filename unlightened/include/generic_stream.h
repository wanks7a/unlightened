#pragma once
#include <type_traits>
#include <vector>

class generic_stream
{
public:
	virtual size_t write(const char* ptr, size_t bytes) = 0;
	virtual size_t read(char* buff, size_t bytes) = 0;
	virtual size_t peek(char* buff, size_t bytes) const = 0;
	virtual ~generic_stream() = default;
};
