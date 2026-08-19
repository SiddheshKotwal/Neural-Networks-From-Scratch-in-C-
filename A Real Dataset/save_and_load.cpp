#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using LayerParameters = std::tuple<Matrix, Matrix>;
using ModelParameters = std::vector<LayerParameters>;

namespace {

void write_matrix(std::ofstream& out, const Matrix& matrix) {
    const std::uint64_t rows = matrix.size();
    const std::uint64_t cols = rows == 0 ? 0 : matrix.front().size();

    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    for (const auto& row : matrix) {
        if (row.size() != cols) {
            throw std::runtime_error("Cannot save a ragged matrix");
        }

        if (cols > 0) {
            out.write(
                reinterpret_cast<const char*>(row.data()),
                static_cast<std::streamsize>(cols * sizeof(double))
            );
        }
    }
}

Matrix read_matrix(std::ifstream& in) {
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;

    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    if (!in) {
        throw std::runtime_error("Invalid model parameter file");
    }

    Matrix matrix(
        static_cast<std::size_t>(rows),
        std::vector<double>(static_cast<std::size_t>(cols))
    );

    for (auto& row : matrix) {
        if (cols > 0) {
            in.read(
                reinterpret_cast<char*>(row.data()),
                static_cast<std::streamsize>(cols * sizeof(double))
            );
        }
    }

    if (!in) {
        throw std::runtime_error(
            "Unexpected end of model parameter file"
        );
    }

    return matrix;
}

} // namespace

void save_parameters(
    const ModelParameters& data,
    const std::string& filename
) {
    std::ofstream out(filename, std::ios::binary);

    if (!out) {
        throw std::runtime_error(
            "Unable to open parameter file for writing: " + filename
        );
    }

    const std::uint64_t layer_count = data.size();

    out.write(
        reinterpret_cast<const char*>(&layer_count),
        sizeof(layer_count)
    );

    for (const auto& parameters : data) {
        write_matrix(out, std::get<0>(parameters));
        write_matrix(out, std::get<1>(parameters));
    }
}

void load_parameters(
    ModelParameters& data,
    const std::string& filename
) {
    std::ifstream in(filename, std::ios::binary);

    if (!in) {
        throw std::runtime_error(
            "Unable to open model parameter file for reading: " + filename
        );
    }

    std::uint64_t layer_count = 0;

    in.read(
        reinterpret_cast<char*>(&layer_count),
        sizeof(layer_count)
    );

    if (!in) {
        throw std::runtime_error("Invalid model parameter file");
    }

    data.clear();
    data.reserve(static_cast<std::size_t>(layer_count));

    for (std::uint64_t i = 0; i < layer_count; ++i) {
        Matrix weights = read_matrix(in);
        Matrix biases = read_matrix(in);

        data.emplace_back(
            std::move(weights),
            std::move(biases)
        );
    }
}
