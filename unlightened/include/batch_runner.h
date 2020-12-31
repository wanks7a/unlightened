#pragma once
#include <functional>

class batch_runner
{
	size_t current_epoch;
	size_t checkpoint_save_time;

	template <typename Data>
	void train(const Data& data, size_t epochs)
	{
		current_epoch = 0;
		for (size_t e = 0; e < epochs; e++)
		{
			const auto& it = data.begin();
			while (it != data.end())
			{

			}
		}
	}
};