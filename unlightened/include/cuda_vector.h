#pragma once
#include <device_vector.h>
#include <cuda_device.h>

class cuda_floats : public device_vector<cuda_device, float>
{
public:
	void operator+=(float val);
	void operator*=(float val);
	void operator-=(float val);
	void operator/=(float val);
	void operator+=(const cuda_floats& vals);
	void operator*=(const cuda_floats& vals);
	void operator-=(const cuda_floats& vals);
	void operator/=(const cuda_floats& vals);
	cuda_floats operator+(float val) const;
	cuda_floats operator*(float val) const;
	cuda_floats operator-(float val) const;
	cuda_floats operator/(float val) const;
	cuda_floats operator+(const cuda_floats& vals) const;
	cuda_floats operator*(const cuda_floats& vals) const;
	cuda_floats operator-(const cuda_floats& vals) const;
	cuda_floats operator/(const cuda_floats& vals);
	cuda_floats& operator=(cuda_floats&& value) = default;
	void sqrt();
	cuda_floats(cuda_floats&& values) = default;
	cuda_floats() = default;
	cuda_floats(const cuda_floats&) = delete;
	cuda_floats& operator=(const cuda_floats&) = delete;
	~cuda_floats() = default;
};