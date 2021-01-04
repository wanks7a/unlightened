#pragma once
#include <generic_stream.h>
#include <string>

class read_write_stream : public generic_stream
{
public:
	read_write_stream(const char* path);
	virtual size_t write(const char* ptr, size_t bytes) override;
	virtual size_t read(char* buff, size_t bytes) override;
	virtual size_t peek(char* buff, size_t bytes) const override;
	virtual bool is_open() const override;
	virtual ~read_write_stream();
private:
	FILE* f_ptr;
	std::string f_path;
};