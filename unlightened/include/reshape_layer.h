#include <serializable_interface.h>

class reshape_layer : public serializable_layer<reshape_layer>
{
    const float* output;
    const float* derivative;
public:
    reshape_layer() = default;
    reshape_layer(shape out_shape);
    void init(const shape& input) override;
    void forward_pass(Layer* prevLayer) override;
    void backprop(Layer* layer) override;
    const float* get_output() const override;
    const float* derivative_wr_to_input() const override;

    template <typename Serializer>
    void serialize_members(Serializer& s) const
    {
    }

    template <typename Serializer>
    void deserialize_members(Serializer& s)
    {
    }
};