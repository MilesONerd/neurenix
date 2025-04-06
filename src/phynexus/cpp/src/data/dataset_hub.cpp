/**
 * @file dataset_hub.cpp
 * @brief DatasetHub implementation for easy dataset loading and management
 */

#include "data/dataset_hub.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <filesystem>
#include <regex>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace phynexus {
namespace data {

static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    auto* mem = static_cast<std::string*>(userp);
    mem->append(static_cast<char*>(contents), realsize);
    return realsize;
}

DatasetFormat format_from_string(const std::string& format_str) {
    std::string lower_format = format_str;
    std::transform(lower_format.begin(), lower_format.end(), lower_format.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    
    if (lower_format == "csv") return DatasetFormat::CSV;
    if (lower_format == "json") return DatasetFormat::JSON;
    if (lower_format == "numpy" || lower_format == "npy" || lower_format == "npz") return DatasetFormat::NUMPY;
    if (lower_format == "pickle" || lower_format == "pkl") return DatasetFormat::PICKLE;
    if (lower_format == "text" || lower_format == "txt") return DatasetFormat::TEXT;
    if (lower_format == "image" || lower_format == "img") return DatasetFormat::IMAGE;
    if (lower_format == "audio") return DatasetFormat::AUDIO;
    if (lower_format == "video") return DatasetFormat::VIDEO;
    if (lower_format == "custom") return DatasetFormat::CUSTOM;
    
    return DatasetFormat::CUSTOM;
}

DatasetFormat format_from_extension(const std::string& path) {
    fs::path file_path(path);
    std::string extension = file_path.extension().string();
    
    if (!extension.empty() && extension[0] == '.') {
        extension = extension.substr(1);
    }
    std::transform(extension.begin(), extension.end(), extension.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    
    if (extension == "csv" || extension == "tsv") return DatasetFormat::CSV;
    if (extension == "json" || extension == "jsonl") return DatasetFormat::JSON;
    if (extension == "npy" || extension == "npz") return DatasetFormat::NUMPY;
    if (extension == "pkl" || extension == "pickle") return DatasetFormat::PICKLE;
    if (extension == "txt" || extension == "text") return DatasetFormat::TEXT;
    if (extension == "jpg" || extension == "jpeg" || extension == "png" || extension == "bmp" || extension == "gif") return DatasetFormat::IMAGE;
    if (extension == "wav" || extension == "mp3" || extension == "ogg" || extension == "flac") return DatasetFormat::AUDIO;
    if (extension == "mp4" || extension == "avi" || extension == "mov" || extension == "mkv") return DatasetFormat::VIDEO;
    
    return DatasetFormat::CUSTOM;
}

Dataset::Dataset(
    DataVariant data,
    DatasetFormat format,
    const std::string& name,
    const std::unordered_map<std::string, std::string>& metadata
) : data_(std::move(data)), format_(format), name_(name), metadata_(metadata) {}

DatasetFormat Dataset::format() const {
    return format_;
}

const std::string& Dataset::name() const {
    return name_;
}

const std::unordered_map<std::string, std::string>& Dataset::metadata() const {
    return metadata_;
}

const Dataset::DataVariant& Dataset::data() const {
    return data_;
}

const std::vector<std::vector<std::string>>& Dataset::as_table() const {
    if (!std::holds_alternative<std::vector<std::vector<std::string>>>(data_)) {
        throw std::runtime_error("Dataset does not contain table data");
    }
    return std::get<std::vector<std::vector<std::string>>>(data_);
}

const std::string& Dataset::as_text() const {
    if (!std::holds_alternative<std::string>(data_)) {
        throw std::runtime_error("Dataset does not contain text data");
    }
    return std::get<std::string>(data_);
}

const std::vector<uint8_t>& Dataset::as_binary() const {
    if (!std::holds_alternative<std::vector<uint8_t>>(data_)) {
        throw std::runtime_error("Dataset does not contain binary data");
    }
    return std::get<std::vector<uint8_t>>(data_);
}

const std::vector<float>& Dataset::as_numeric() const {
    if (!std::holds_alternative<std::vector<float>>(data_)) {
        throw std::runtime_error("Dataset does not contain numeric data");
    }
    return std::get<std::vector<float>>(data_);
}

std::pair<Dataset, Dataset> Dataset::split(
    float ratio,
    bool shuffle,
    std::optional<int> seed
) const {
    if (ratio <= 0.0f || ratio >= 1.0f) {
        throw std::invalid_argument("Split ratio must be between 0.0 and 1.0");
    }
    
    std::mt19937 rng;
    if (seed) {
        rng.seed(*seed);
    } else {
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    
    if (std::holds_alternative<std::vector<std::vector<std::string>>>(data_)) {
        const auto& table = std::get<std::vector<std::vector<std::string>>>(data_);
        size_t n = table.size();
        if (n == 0) {
            throw std::runtime_error("Cannot split empty dataset");
        }
        
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        
        size_t split_idx = static_cast<size_t>(n * ratio);
        
        std::vector<std::vector<std::string>> train_data;
        std::vector<std::vector<std::string>> val_data;
        
        train_data.reserve(split_idx);
        val_data.reserve(n - split_idx);
        
        for (size_t i = 0; i < split_idx; ++i) {
            train_data.push_back(table[indices[i]]);
        }
        
        for (size_t i = split_idx; i < n; ++i) {
            val_data.push_back(table[indices[i]]);
        }
        
        return {
            Dataset(std::move(train_data), format_, name_ + "_train", metadata_),
            Dataset(std::move(val_data), format_, name_ + "_val", metadata_)
        };
    } else if (std::holds_alternative<std::vector<float>>(data_)) {
        const auto& numeric = std::get<std::vector<float>>(data_);
        size_t n = numeric.size();
        if (n == 0) {
            throw std::runtime_error("Cannot split empty dataset");
        }
        
        std::vector<size_t> shape;
        if (metadata_.count("shape")) {
            std::string shape_str = metadata_.at("shape");
            std::stringstream ss(shape_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                shape.push_back(std::stoul(item));
            }
        }
        
        if (shape.empty()) {
            shape.push_back(n);
        }
        
        size_t sample_size = 1;
        for (size_t i = 1; i < shape.size(); ++i) {
            sample_size *= shape[i];
        }
        
        size_t num_samples = shape[0];
        
        std::vector<size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        
        size_t split_idx = static_cast<size_t>(num_samples * ratio);
        
        std::vector<float> train_data;
        std::vector<float> val_data;
        
        train_data.reserve(split_idx * sample_size);
        val_data.reserve((num_samples - split_idx) * sample_size);
        
        for (size_t i = 0; i < split_idx; ++i) {
            size_t offset = indices[i] * sample_size;
            train_data.insert(train_data.end(), 
                             numeric.begin() + offset, 
                             numeric.begin() + offset + sample_size);
        }
        
        for (size_t i = split_idx; i < num_samples; ++i) {
            size_t offset = indices[i] * sample_size;
            val_data.insert(val_data.end(), 
                           numeric.begin() + offset, 
                           numeric.begin() + offset + sample_size);
        }
        
        auto train_metadata = metadata_;
        auto val_metadata = metadata_;
        
        if (!shape.empty()) {
            shape[0] = split_idx;
            std::stringstream train_ss;
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) train_ss << ",";
                train_ss << shape[i];
            }
            train_metadata["shape"] = train_ss.str();
            
            shape[0] = num_samples - split_idx;
            std::stringstream val_ss;
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) val_ss << ",";
                val_ss << shape[i];
            }
            val_metadata["shape"] = val_ss.str();
        }
        
        return {
            Dataset(std::move(train_data), format_, name_ + "_train", train_metadata),
            Dataset(std::move(val_data), format_, name_ + "_val", val_metadata)
        };
    } else {
        throw std::runtime_error("Split is only supported for table and numeric data");
    }
}

DatasetHub::DatasetHub(const std::string& cache_dir) {
    if (cache_dir.empty()) {
        const char* home_dir = std::getenv("HOME");
        if (home_dir) {
            cache_dir_ = std::string(home_dir) + "/.neurenix/datasets";
        } else {
            cache_dir_ = "./.neurenix/datasets";
        }
    } else {
        cache_dir_ = cache_dir;
    }
    
    fs::create_directories(cache_dir_);
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

void DatasetHub::register_dataset(
    const std::string& name,
    const std::string& url,
    std::optional<DatasetFormat> format,
    const std::unordered_map<std::string, std::string>& metadata
) {
    RegisteredDataset dataset;
    dataset.url = url;
    dataset.format = format;
    dataset.metadata = metadata;
    
    registered_datasets_[name] = std::move(dataset);
}

Dataset DatasetHub::load_dataset(
    const std::string& source,
    std::optional<DatasetFormat> format,
    bool force_download,
    const std::unordered_map<std::string, std::string>& options
) {
    std::string url;
    std::unordered_map<std::string, std::string> metadata;
    
    if (registered_datasets_.count(source)) {
        const auto& dataset = registered_datasets_[source];
        url = dataset.url;
        format = format.has_value() ? format : dataset.format;
        metadata = dataset.metadata;
    } else {
        url = source;
    }
    
    bool is_remote = url.find("http://") == 0 || url.find("https://") == 0 || url.find("ftp://") == 0;
    
    std::string local_path;
    if (is_remote) {
        local_path = download_dataset(url, force_download);
    } else {
        local_path = url;
    }
    
    DatasetFormat actual_format;
    if (format) {
        actual_format = *format;
    } else {
        actual_format = format_from_extension(local_path);
    }
    
    auto data = load_data(local_path, actual_format, options);
    
    return Dataset(std::move(data), actual_format, source, metadata);
}

std::string DatasetHub::download_dataset(const std::string& url, bool force_download) {
    std::string filename;
    
    size_t last_slash = url.find_last_of('/');
    if (last_slash != std::string::npos && last_slash < url.length() - 1) {
        filename = url.substr(last_slash + 1);
        
        size_t query_start = filename.find('?');
        if (query_start != std::string::npos) {
            filename = filename.substr(0, query_start);
        }
    }
    
    if (filename.empty()) {
        std::hash<std::string> hasher;
        size_t hash = hasher(url);
        filename = "dataset_" + std::to_string(hash);
    }
    
    std::string local_path = cache_dir_ + "/" + filename;
    
    if (!force_download && fs::exists(local_path)) {
        return local_path;
    }
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    std::string temp_path = local_path + ".tmp";
    FILE* fp = fopen(temp_path.c_str(), "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to create file: " + temp_path);
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    
    CURLcode res = curl_easy_perform(curl);
    
    fclose(fp);
    
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        fs::remove(temp_path);
        throw std::runtime_error("Failed to download dataset: " + std::string(curl_easy_strerror(res)));
    }
    
    curl_easy_cleanup(curl);
    
    fs::rename(temp_path, local_path);
    
    return local_path;
}

Dataset::DataVariant DatasetHub::load_data(
    const std::string& path,
    DatasetFormat format,
    const std::unordered_map<std::string, std::string>& options
) {
    switch (format) {
        case DatasetFormat::CSV:
            return load_csv(path, options);
        case DatasetFormat::JSON:
            return load_json(path, options);
        case DatasetFormat::NUMPY:
            return load_numpy(path, options);
        case DatasetFormat::TEXT:
            return load_text(path, options);
        case DatasetFormat::IMAGE:
            return load_image(path, options);
        case DatasetFormat::AUDIO:
            return load_audio(path, options);
        case DatasetFormat::VIDEO:
            return load_video(path, options);
        default:
            throw std::runtime_error("Unsupported format");
    }
}

Dataset::DataVariant DatasetHub::load_csv(
    const std::string& path,
    const std::unordered_map<std::string, std::string>& options
) {
    char delimiter = ',';
    bool has_header = true;
    
    if (options.count("delimiter")) {
        delimiter = options.at("delimiter")[0];
    }
    
    if (options.count("has_header")) {
        has_header = options.at("has_header") == "true";
    }
    
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::vector<std::vector<std::string>> data;
    std::string line;
    
    if (has_header && std::getline(file, line)) {
        if (options.count("include_header") && options.at("include_header") == "true") {
            std::vector<std::string> header;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, delimiter)) {
                cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                
                header.push_back(cell);
            }
            
            data.push_back(header);
        }
    }
    
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, delimiter)) {
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            
            row.push_back(cell);
        }
        
        data.push_back(row);
    }
    
    return data;
}

Dataset::DataVariant DatasetHub::load_json(
    const std::string& path,
    const std::unordered_map<std::string, std::string>& options
) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    return json_str;
}

Dataset::DataVariant DatasetHub::load_numpy(
    const std::string& path,
    const std::unordered_map<std::string, std::string>& options
) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    return data;
}

Dataset::DataVariant DatasetHub::load_text(
    const std::string& path,
    const std::unordered_map<std::string, std::string>& options
) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    return text;
}

Dataset::DataVariant DatasetHub::load_image(
    const std::string& path,
    const std::unordered_map<std::string, std::string>& options
) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    return data;
}

Dataset::DataVariant DatasetHub::load_audio(
    const std::string& path,
    const std::unordered_map<std::string, std::string>& options
) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    return data;
}

Dataset::DataVariant DatasetHub::load_video(
    const std::string& path,
    const std::unordered_map<std::string, std::string>& options
) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    return data;
}

} // namespace data
} // namespace phynexus
