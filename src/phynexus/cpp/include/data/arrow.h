/**
 * @file arrow.h
 * @brief Apache Arrow integration for Phynexus.
 * 
 * This header provides functionality for working with Apache Arrow data structures,
 * enabling efficient in-memory data processing and interoperability with other
 * data processing frameworks.
 */

#ifndef PHYNEXUS_DATA_ARROW_H
#define PHYNEXUS_DATA_ARROW_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <variant>

#include "phynexus/tensor.h"
#include "phynexus/error.h"

namespace phynexus {
namespace data {

/**
 * @brief Wrapper for Arrow data structures with integration to Phynexus tensors.
 */
class ArrowTable {
public:
    /**
     * @brief Default constructor.
     */
    ArrowTable();

    /**
     * @brief Construct from raw data.
     * 
     * @param data Map of column names to data vectors
     */
    template <typename T>
    ArrowTable(const std::unordered_map<std::string, std::vector<T>>& data);

    /**
     * @brief Destructor.
     */
    ~ArrowTable();

    /**
     * @brief Get the number of rows in the table.
     * 
     * @return Number of rows
     */
    size_t num_rows() const;

    /**
     * @brief Get the number of columns in the table.
     * 
     * @return Number of columns
     */
    size_t num_columns() const;

    /**
     * @brief Get the column names in the table.
     * 
     * @return Column names
     */
    std::vector<std::string> column_names() const;

    /**
     * @brief Convert a column to a Phynexus Tensor.
     * 
     * @param column Column name or index
     * @return Tensor
     */
    Tensor to_tensor(const std::variant<std::string, size_t>& column) const;

    /**
     * @brief Convert all columns to Phynexus Tensors.
     * 
     * @return Map of column names to Tensors
     */
    std::unordered_map<std::string, Tensor> to_tensors() const;

    /**
     * @brief Create an ArrowTable from a Phynexus Tensor.
     * 
     * @param tensor Tensor
     * @param name Column name
     * @return ArrowTable
     */
    static ArrowTable from_tensor(const Tensor& tensor, const std::string& name = "data");

    /**
     * @brief Create an ArrowTable from a map of Phynexus Tensors.
     * 
     * @param tensors Map of column names to Tensors
     * @return ArrowTable
     */
    static ArrowTable from_tensors(const std::unordered_map<std::string, Tensor>& tensors);

    /**
     * @brief Convert to a list of dictionaries.
     * 
     * @return List of dictionaries
     */
    std::vector<std::unordered_map<std::string, std::variant<int, float, double, bool, std::string>>> to_records() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Read a Parquet file into an ArrowTable.
 * 
 * @param path Path to the Parquet file
 * @param columns Columns to read
 * @return ArrowTable
 */
ArrowTable read_parquet(const std::string& path, const std::vector<std::string>& columns = {});

/**
 * @brief Write an ArrowTable to a Parquet file.
 * 
 * @param table ArrowTable
 * @param path Path to write the Parquet file
 * @param compression Compression algorithm
 * @param row_group_size Row group size
 * @param version Parquet format version
 * @param write_statistics Whether to write statistics
 */
void write_parquet(
    const ArrowTable& table,
    const std::string& path,
    const std::string& compression = "snappy",
    std::optional<size_t> row_group_size = std::nullopt,
    const std::string& version = "2.0",
    bool write_statistics = true
);

/**
 * @brief Convert a Phynexus Tensor to an Arrow Array.
 * 
 * @param tensor Tensor
 * @return Arrow Array
 */
void* tensor_to_arrow(const Tensor& tensor);

/**
 * @brief Convert an Arrow Array to a Phynexus Tensor.
 * 
 * @param array Arrow Array
 * @return Tensor
 */
Tensor arrow_to_tensor(void* array);

} // namespace data
} // namespace phynexus

#endif // PHYNEXUS_DATA_ARROW_H
