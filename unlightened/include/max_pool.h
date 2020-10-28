#include <shape.h>
#include <serializable_interface.h>

void max_pooling(const float* input, const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size);

void max_pooling_backprop(const float* input, const shape& input_shape, float* output, const shape& output_shape, char* mask, int filter_size);

class max_pool : public serializable_layer<max_pool>
{
    cuVector<char> mask;
    cuVector<float> output;
    cuVector<float> derivative;
    cuVector<float> input;
    cuVector<float> derivative_input;
    int filter_size;
public:
    max_pool();
    max_pool(int f_size);

    virtual void init(const shape& input) override;
    virtual void forward_pass(Layer* prevLayer) override;
    virtual void backprop(Layer* layer) override;
    virtual const float* get_output() override;
    virtual const float* derivative_wr_to_input() override;

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
        s << filter_size;
    }

    template <typename Serializer>
    void deserialize_members(Serializer& s)
    {
        s >> filter_size;
        init(input_shape);
    }

    ~max_pool() = default;
};