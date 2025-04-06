/**
 * @file dataset_hub.h
 * @brief DatasetHub for easy dataset loading and management
 */

#ifndef PHYNEXUS_DATA_DATASET_HUB_H
#define PHYNEXUS_DATA_DATASET_HUB_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <optional>
#include <variant>

namespace phynexus {
namespace data {

/**
 * @brief Supported dataset formats
 */
enum class DatasetFormat {
    CSV,
    JSON,
    NUMPY,
    PICKLE,
    TEXT,
    IMAGE,
    AUDIO,
    VIDEO,
    CUSTOM
};

/**
 * @brief Convert string to DatasetFormat
 * @param format_str Format string
 * @return DatasetFormat enum value
 */
DatasetFormat format_from_string(const std::string& format_str);

/**
 * @brief Determine format from file extension
 * @param path File path
 * @return DatasetFormat enum value
 */
DatasetFormat format_from_extension(const std::string& path);

/**
 * @brief Dataset class for handling various data formats
 */
class Dataset {
public:
    /**
     * @brief Dataset data types
     */
    using DataVariant = std::variant<
        std::vector<std::vector<std::string>>,  // Table data (CSV)
        std::string,                            // Text data or JSON string
        std::vector<uint8_t>,                   // Binary data (images, audio, etc.)
        std::vector<float>                      // Numeric data (with shape information in metadata)
    >;

    /**
     * @brief Create a new dataset
     * @param data Dataset data
     * @param format Dataset format
     * @param name Dataset name
     * @param metadata Dataset metadata
     */
    Dataset(
        DataVariant data,
        DatasetFormat format,
        const std::string& name = "",
        const std::unordered_map<std::string, std::string>& metadata = {}
    );

    /**
     * @brief Get the dataset format
     * @return Dataset format
     */
    DatasetFormat format() const;

    /**
     * @brief Get the dataset name
     * @return Dataset name
     */
    const std::string& name() const;

    /**
     * @brief Get the dataset metadata
     * @return Dataset metadata
     */
    const std::unordered_map<std::string, std::string>& metadata() const;

    /**
     * @brief Get the dataset data
     * @return Dataset data
     */
    const DataVariant& data() const;

    /**
     * @brief Get the dataset data as a table
     * @return Dataset data as a table
     */
    const std::vector<std::vector<std::string>>& as_table() const;

    /**
     * @brief Get the dataset data as text
     * @return Dataset data as text
     */
    const std::string& as_text() const;

    /**
     * @brief Get the dataset data as binary
     * @return Dataset data as binary
     */
    const std::vector<uint8_t>& as_binary() const;

    /**
     * @brief Get the dataset data as numeric
     * @return Dataset data as numeric
     */
    const std::vector<float>& as_numeric() const;

    /**
     * @brief Split the dataset into training and validation sets
     * @param ratio Proportion of data to use for training (0.0 to 1.0)
     * @param shuffle Whether to shuffle the data before splitting
     * @param seed Random seed for reproducibility
     * @return Pair of (train_dataset, val_dataset)
     */
    std::pair<Dataset, Dataset> split(
        float ratio = 0.8f,
        bool shuffle = true,
        std::optional<int> seed = std::nullopt
    ) const;

private:
    DataVariant data_;
    DatasetFormat format_;
    std::string name_;
    std::unordered_map<std::string, std::string> metadata_;
};

/**
 * @brief DatasetHub for managing and loading datasets
 */
class DatasetHub {
public:
    /**
     * @brief Create a new DatasetHub
     * @param cache_dir Directory to cache downloaded datasets
     */
    explicit DatasetHub(const std::string& cache_dir = "");

    /**
     * @brief Register a dataset with the hub
     * @param name Dataset name
     * @param url Dataset URL or file path
     * @param format Dataset format (auto-detected if nullopt)
     * @param metadata Dataset metadata
     */
    void register_dataset(
        const std::string& name,
        const std::string& url,
        std::optional<DatasetFormat> format = std::nullopt,
        const std::unordered_map<std::string, std::string>& metadata = {}
    );

    /**
     * @brief Load a dataset from a URL or file path
     * @param source Dataset URL, file path, or registered name
     * @param format Dataset format (auto-detected if nullopt)
     * @param force_download Whether to force download even if cached
     * @param options Additional options for specific formats
     * @return Loaded dataset
     */
    Dataset load_dataset(
        const std::string& source,
        std::optional<DatasetFormat> format = std::nullopt,
        bool force_download = false,
        const std::unordered_map<std::string, std::string>& options = {}
    );

private:
    /**
     * @brief Registered dataset information
     */
    struct RegisteredDataset {
        std::string url;
        std::optional<DatasetFormat> format;
        std::unordered_map<std::string, std::string> metadata;
    };

    /**
     * @brief Download a dataset from a URL
     * @param url Dataset URL
     * @param force_download Whether to force download even if cached
     * @return Local path to the downloaded dataset
     */
    std::string download_dataset(const std::string& url, bool force_download);

    /**
     * @brief Load data from a file based on format
     * @param path File path
     * @param format Dataset format
     * @param options Additional options for specific formats
     * @return Dataset data
     */
    Dataset::DataVariant load_data(
        const std::string& path,
        DatasetFormat format,
        const std::unordered_map<std::string, std::string>& options
    );

    /**
     * @brief Load CSV data
     * @param path File path
     * @param options Additional options for CSV loading
     * @return Dataset data
     */
    Dataset::DataVariant load_csv(
        const std::string& path,
        const std::unordered_map<std::string, std::string>& options
    );

    /**
     * @brief Load JSON data
     * @param path File path
     * @param options Additional options for JSON loading
     * @return Dataset data
     */
    Dataset::DataVariant load_json(
        const std::string& path,
        const std::unordered_map<std::string, std::string>& options
    );

    /**
     * @brief Load NumPy data
     * @param path File path
     * @param options Additional options for NumPy loading
     * @return Dataset data
     */
    Dataset::DataVariant load_numpy(
        const std::string& path,
        const std::unordered_map<std::string, std::string>& options
    );

    /**
     * @brief Load text data
     * @param path File path
     * @param options Additional options for text loading
     * @return Dataset data
     */
    Dataset::DataVariant load_text(
        const std::string& path,
        const std::unordered_map<std::string, std::string>& options
    );

    /**
     * @brief Load image data
     * @param path File path
     * @param options Additional options for image loading
     * @return Dataset data
     */
    Dataset::DataVariant load_image(
        const std::string& path,
        const std::unordered_map<std::string, std::string>& options
    );

    /**
     * @brief Load audio data
     * @param path File path
     * @param options Additional options for audio loading
     * @return Dataset data
     */
    Dataset::DataVariant load_audio(
        const std::string& path,
        const std::unordered_map<std::string, std::string>& options
    );

    /**
     * @brief Load video data
     * @param path File path
     * @param options Additional options for video loading
     * @return Dataset data
     */
    Dataset::DataVariant load_video(
        const std::string& path,
        const std::unordered_map<std::string, std::string>& options
    );

    std::string cache_dir_;
    std::unordered_map<std::string, RegisteredDataset> registered_datasets_;
};

} // namespace data
} // namespace phynexus

#endif // PHYNEXUS_DATA_DATASET_HUB_H
