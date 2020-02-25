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
                row<std::string> temp_row;
                temp_row.elements = std::move(get_line_tokens(file));
                csv_rows.emplace_back(std::move(temp_row));
            }
            file.close();
            csv_rows.erase(csv_rows.begin() + csv_rows.size() - 1);
            convert_csv_rows(csv_rows);
        }
    }

private:
    std::vector<std::string> get_line_tokens(std::istream& str)
    {
        std::vector<std::string>   result;
        std::string                line;
        std::getline(str, line);

        std::stringstream          lineStream(line);
        std::string                cell;

        while (std::getline(lineStream, cell, ','))
        {
            result.push_back(cell);
        }

        if (!lineStream && cell.empty())
        {
            result.push_back("");
        }
        return result;
    }

    void convert_csv_rows(std::vector<row<std::string>>& csv_rows)
    {
        for (const auto& r : csv_rows)
        {
            row<T> temp_row;
            for (const auto& str : r.elements)
            {
                temp_row.elements.emplace_back(convert(str));
            }
            rows.emplace_back(std::move(temp_row));
        }
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