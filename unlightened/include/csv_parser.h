#include <sstream>
#include <string>
#include <vector>
#include <fstream>

template <typename T>
class csv
{
public:

    template <typename ROW_TYPE>
    struct row
    {
        std::vector<ROW_TYPE> elements;
    };

    std::vector<row<T>> rows;

    csv(const std::string& path)
    {
        std::ifstream file(path.data());
        std::vector<row<std::string>> csv_rows;
        if (file.good())
        {
            while (!file.eof())
            {
                row<T> temp_row;
                temp_row.elements = std::move(get_line_tokens(file));
                rows.emplace_back(std::move(temp_row));
            }
            file.close();
            rows.erase(rows.begin() + rows.size() - 1);
        }
    }

private:
    std::vector<T> get_line_tokens(std::istream& str)
    {
        std::vector<T>             result;
        std::string                line;
        std::getline(str, line);

        std::stringstream          lineStream(line);
        std::string                cell;

        while (std::getline(lineStream, cell, ','))
        {
            result.push_back(convert(cell));
        }

        if (!lineStream && cell.empty())
        {
            result.push_back(T());
        }
        return result;
    }

    template <typename Value = T, std::enable_if_t<std::is_floating_point<Value>::value, void*> = nullptr>
    Value convert(const std::string& val)
    {
        return std::stof(val);
    }

    template <typename Value = T, std::enable_if_t<std::is_integral<Value>::value, void*> = nullptr>
    Value convert(const std::string& val)
    {
        return std::stoi(val);
    }
};