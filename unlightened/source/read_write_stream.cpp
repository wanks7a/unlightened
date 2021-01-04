#include <read_write_stream.h>

read_write_stream::read_write_stream(const char* path) : f_path(path), f_ptr(nullptr)
{
	f_ptr = fopen(path, "rb+");
	if (!f_ptr)
		f_ptr = fopen(path, "wb+");
}

size_t read_write_stream::write(const char* ptr, size_t bytes)
{
	size_t result = fwrite(ptr, sizeof(char), bytes, f_ptr);
	if (bytes != result)
		throw("read_write_stream::write");
	return result;
}

size_t read_write_stream::read(char* buff, size_t bytes)
{
	size_t result = fread(buff, sizeof(char), bytes, f_ptr);
	if (bytes != result)
		throw("read_write_stream::read");
	return result;
}

size_t read_write_stream::peek(char* buff, size_t bytes) const
{
	size_t result = fread(buff, sizeof(char), bytes, f_ptr);
	if (bytes != result)
		throw("read_write_stream::read");
	fseek(f_ptr, -static_cast<long>(result), SEEK_CUR);
	return result;
}

bool read_write_stream::is_open() const
{
	return f_ptr != nullptr;
}

read_write_stream::~read_write_stream()
{
	if (f_ptr)
		fclose(f_ptr);
}