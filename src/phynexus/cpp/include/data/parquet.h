/**
 * @file parquet.h
 * @brief Parquet integration for Phynexus.
 * 
 * This header provides functionality for working with Parquet files,
 * enabling efficient storage and retrieval of structured data.
 */

#ifndef PHYNEXUS_DATA_PARQUET_H
#define PHYNEXUS_DATA_PARQUET_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <variant>

#include "data/arrow.h"
#include "phynexus/tensor.h"
#include "phynexus/error.h"

namespace phynexus {
namespace data {

/**
 * @brief Dataset for reading and writing Parquet files.
 */
class ParquetDataset {
public:
    /**
     * @brief Default constructor.
     */
    ParquetDataset();

    /**
     * @brief Construct from a Parquet file.
     * 
     * @param path Path to the Parquet file or directory
     * @param columns Columns to read
     */
    ParquetDataset(const std::string& path, const std::vector<std::string>& columns = {});

    /**
     * @brief Destructor.
     */
    ~ParquetDataset();

    /**
     * @brief Get the number of rows in the dataset.
     * 
     * @return Number of rows
     */
    size_t num_rows() const;

    /**
     * @brief Get the number of columns in the dataset.
     * 
     * @return Number of columns
     */
    size_t num_columns() const;

    /**
     * @brief Get the column names in the dataset.
     * 
     * @return Column names
     */
    std::vector<std::string> column_names() const;

    /**
     * @brief Read the dataset into an ArrowTable.
     * 
     * @param columns Columns to read
     * @param batch_size Batch size for reading
     * @return ArrowTable
     */
    ArrowTable read(const std::vector<std::string>& columns = {}, std::optional<size_t> batch_size = std::nullopt) const;

    /**
     * @brief Read a specific row group from the dataset.
     * 
     * @param row_group_index Row group index
     * @param columns Columns to read
     * @return ArrowTable
     */
    ArrowTable read_row_group(size_t row_group_index, const std::vector<std::string>& columns = {}) const;

    /**
     * @brief Read specific row groups from the dataset.
     * 
     * @param row_group_indices Row group indices
     * @param columns Columns to read
     * @return ArrowTable
     */
    ArrowTable read_row_groups(const std::vector<size_t>& row_group_indices, const std::vector<std::string>& columns = {}) const;

    /**
     * @brief Convert the dataset to Phynexus Tensors.
     * 
     * @param columns Columns to convert
     * @return Map of column names to Tensors
     */
    std::unordered_map<std::string, Tensor> to_tensors(const std::vector<std::string>& columns = {}) const;

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
    static void write(
        const ArrowTable& table,
        const std::string& path,
        const std::string& compression = "snappy",
        std::optional<size_t> row_group_size = std::nullopt,
        const std::string& version = "2.0",
        bool write_statistics = true
    );

    /**
     * @brief Write an ArrowTable to a partitioned Parquet dataset.
     * 
     * @param table ArrowTable
     * @param root_path Root path for the dataset
     * @param partition_cols Columns to partition by
     * @param compression Compression algorithm
     */
    static void write_to_dataset(
        const ArrowTable& table,
        const std::string& root_path,
        const std::vector<std::string>& partition_cols = {},
        const std::string& compression = "snappy"
    );

    /**
     * @brief Create a ParquetDataset from a dictionary of Phynexus Tensors.
     * 
     * @param tensors Map of column names to Tensors
     * @return ParquetDataset
     */
    static ArrowTable from_tensors(const std::unordered_map<std::string, Tensor>& tensors);

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
 * @brief Write an ArrowTable to a partitioned Parquet dataset.
 * 
 * @param table ArrowTable
 * @param root_path Root path for the dataset
 * @param partition_cols Columns to partition by
 * @param compression Compression algorithm
 */
void write_to_dataset(
    const ArrowTable& table,
    const std::string& root_path,
    const std::vector<std::string>& partition_cols = {},
    const std::string& compression = "snappy"
);

} // namespace data
} // namespace phynexus

#endif // PHYNEXUS_DATA_PARQUET_H
