
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList, PyString, PyBytes};
use pyo3::exceptions::PyValueError;

use ndarray::{Array, ArrayD};
use serde_json::{Value, from_str, from_reader};
use csv::ReaderBuilder;

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFormat {
    CSV,
    JSON,
    NUMPY,
    PICKLE,
    TEXT,
    IMAGE,
    AUDIO,
    VIDEO,
    CUSTOM,
}

impl DatasetFormat {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "csv" => Ok(DatasetFormat::CSV),
            "json" => Ok(DatasetFormat::JSON),
            "numpy" | "npy" | "npz" => Ok(DatasetFormat::NUMPY),
            "pickle" | "pkl" => Ok(DatasetFormat::PICKLE),
            "text" | "txt" => Ok(DatasetFormat::TEXT),
            "image" | "img" => Ok(DatasetFormat::IMAGE),
            "audio" => Ok(DatasetFormat::AUDIO),
            "video" => Ok(DatasetFormat::VIDEO),
            "custom" => Ok(DatasetFormat::CUSTOM),
            _ => Err(PhynexusError::InvalidArgument(format!("Unsupported format: {}", s))),
        }
    }
    
    pub fn from_extension(path: &Path) -> Result<Self> {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
            
        match extension.as_str() {
            "csv" | "tsv" => Ok(DatasetFormat::CSV),
            "json" | "jsonl" => Ok(DatasetFormat::JSON),
            "npy" | "npz" => Ok(DatasetFormat::NUMPY),
            "pkl" | "pickle" => Ok(DatasetFormat::PICKLE),
            "txt" | "text" => Ok(DatasetFormat::TEXT),
            "jpg" | "jpeg" | "png" | "bmp" | "gif" => Ok(DatasetFormat::IMAGE),
            "wav" | "mp3" | "ogg" | "flac" => Ok(DatasetFormat::AUDIO),
            "mp4" | "avi" | "mov" | "mkv" => Ok(DatasetFormat::VIDEO),
            _ => Ok(DatasetFormat::CUSTOM),
        }
    }
}

pub struct Dataset {
    data: Arc<Mutex<DatasetData>>,
    format: DatasetFormat,
    name: Option<String>,
    metadata: HashMap<String, String>,
}

enum DatasetData {
    Tensor(Tensor),
    Array(ArrayD<f32>),
    Text(String),
    Bytes(Vec<u8>),
    Json(Value),
    Table(Vec<Vec<String>>),
}

impl Dataset {
    pub fn new(
        data: DatasetData,
        format: DatasetFormat,
        name: Option<String>,
        metadata: HashMap<String, String>,
    ) -> Self {
        Dataset {
            data: Arc::new(Mutex::new(data)),
            format,
            name,
            metadata,
        }
    }
    
    pub fn format(&self) -> DatasetFormat {
        self.format
    }
    
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
    
    pub fn to_tensor(&self) -> Result<Tensor> {
        let data = self.data.lock().unwrap();
        match &*data {
            DatasetData::Tensor(tensor) => Ok(tensor.clone()),
            DatasetData::Array(array) => {
                let tensor = Tensor::from_array(array.clone())?;
                Ok(tensor)
            },
            _ => Err(PhynexusError::InvalidOperation(
                "Cannot convert this dataset type to tensor".to_string()
            )),
        }
    }
}

pub struct DatasetHub {
    cache_dir: PathBuf,
    registered_datasets: HashMap<String, RegisteredDataset>,
}

struct RegisteredDataset {
    url: String,
    format: DatasetFormat,
    metadata: HashMap<String, String>,
}

impl DatasetHub {
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            home.join(".neurenix").join("datasets")
        });
        
        std::fs::create_dir_all(&cache_dir).unwrap_or_else(|e| {
            eprintln!("Warning: Failed to create cache directory: {}", e);
        });
        
        DatasetHub {
            cache_dir,
            registered_datasets: HashMap::new(),
        }
    }
    
    pub fn register_dataset(
        &mut self,
        name: &str,
        url: &str,
        format: Option<DatasetFormat>,
        metadata: Option<HashMap<String, String>>,
    ) {
        let format = format.unwrap_or_else(|| {
            let path = Path::new(url);
            DatasetFormat::from_extension(path).unwrap_or(DatasetFormat::CUSTOM)
        });
        
        let metadata = metadata.unwrap_or_default();
        
        self.registered_datasets.insert(
            name.to_string(),
            RegisteredDataset {
                url: url.to_string(),
                format,
                metadata,
            },
        );
    }
    
    pub fn load_dataset(
        &self,
        source: &str,
        format: Option<DatasetFormat>,
        force_download: bool,
        options: &str,
    ) -> Result<Dataset> {
        let options: HashMap<String, Value> = from_str(options)
            .map_err(|e| PhynexusError::InvalidArgument(format!("Invalid options JSON: {}", e)))?;
        
        let (url, format, metadata) = if let Some(dataset) = self.registered_datasets.get(source) {
            (
                dataset.url.clone(),
                format.unwrap_or(dataset.format),
                dataset.metadata.clone(),
            )
        } else {
            (
                source.to_string(),
                format.unwrap_or_else(|| {
                    let path = Path::new(source);
                    DatasetFormat::from_extension(path).unwrap_or(DatasetFormat::CUSTOM)
                }),
                HashMap::new(),
            )
        };
        
        let is_remote = url.starts_with("http://") || url.starts_with("https://") || url.starts_with("ftp://");
        
        let local_path = if is_remote {
            self.download_dataset(&url, force_download)?
        } else {
            PathBuf::from(url)
        };
        
        let data = self.load_data(&local_path, format, &options)?;
        
        Ok(Dataset::new(
            data,
            format,
            Some(source.to_string()),
            metadata,
        ))
    }
    
    fn download_dataset(&self, url: &str, force_download: bool) -> Result<PathBuf> {
        use std::fs::File;
        use std::io::Write;
        use url::Url;
        use md5::{Md5, Digest};
        
        let url_parsed = Url::parse(url)
            .map_err(|e| PhynexusError::InvalidArgument(format!("Invalid URL: {}", e)))?;
            
        let filename = url_parsed.path_segments()
            .and_then(|segments| segments.last())
            .unwrap_or("");
            
        let filename = if filename.is_empty() {
            let mut hasher = Md5::new();
            hasher.update(url.as_bytes());
            let hash = hasher.finalize();
            format!("dataset_{:x}", hash)
        } else {
            filename.to_string()
        };
        
        let local_path = self.cache_dir.join(filename);
        
        if !local_path.exists() || force_download {
            println!("Downloading dataset from {}", url);
            
            let temp_path = local_path.with_extension("tmp");
            let mut file = File::create(&temp_path)
                .map_err(|e| PhynexusError::IoError(format!("Failed to create file: {}", e)))?;
                
            let response = ureq::get(url)
                .call()
                .map_err(|e| PhynexusError::IoError(format!("Failed to download dataset: {}", e)))?;
                
            let mut reader = response.into_reader();
            let mut buffer = [0; 8192];
            
            loop {
                let bytes_read = reader.read(&mut buffer)
                    .map_err(|e| PhynexusError::IoError(format!("Failed to read response: {}", e)))?;
                    
                if bytes_read == 0 {
                    break;
                }
                
                file.write_all(&buffer[..bytes_read])
                    .map_err(|e| PhynexusError::IoError(format!("Failed to write to file: {}", e)))?;
            }
            
            std::fs::rename(&temp_path, &local_path)
                .map_err(|e| PhynexusError::IoError(format!("Failed to rename file: {}", e)))?;
        }
        
        Ok(local_path)
    }
    
    fn load_data(&self, path: &Path, format: DatasetFormat, options: &HashMap<String, Value>) -> Result<DatasetData> {
        match format {
            DatasetFormat::CSV => self.load_csv(path, options),
            DatasetFormat::JSON => self.load_json(path, options),
            DatasetFormat::NUMPY => self.load_numpy(path, options),
            DatasetFormat::TEXT => self.load_text(path, options),
            DatasetFormat::IMAGE => self.load_image(path, options),
            DatasetFormat::AUDIO => self.load_audio(path, options),
            DatasetFormat::VIDEO => self.load_video(path, options),
            _ => Err(PhynexusError::InvalidArgument(format!("Unsupported format: {:?}", format))),
        }
    }
    
    fn load_csv(&self, path: &Path, options: &HashMap<String, Value>) -> Result<DatasetData> {
        let file = File::open(path)
            .map_err(|e| PhynexusError::IoError(format!("Failed to open file: {}", e)))?;
            
        let mut reader = ReaderBuilder::new()
            .has_headers(options.get("has_headers").and_then(|v| v.as_bool()).unwrap_or(true))
            .delimiter(options.get("delimiter").and_then(|v| v.as_str()).unwrap_or(",").as_bytes()[0])
            .from_reader(file);
            
        let mut data = Vec::new();
        for result in reader.records() {
            let record = result.map_err(|e| PhynexusError::IoError(format!("Failed to read CSV record: {}", e)))?;
            let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
            data.push(row);
        }
        
        Ok(DatasetData::Table(data))
    }
    
    fn load_json(&self, path: &Path, _options: &HashMap<String, Value>) -> Result<DatasetData> {
        let file = File::open(path)
            .map_err(|e| PhynexusError::IoError(format!("Failed to open file: {}", e)))?;
            
        let json: Value = from_reader(file)
            .map_err(|e| PhynexusError::IoError(format!("Failed to parse JSON: {}", e)))?;
            
        Ok(DatasetData::Json(json))
    }
    
    fn load_numpy(&self, path: &Path, _options: &HashMap<String, Value>) -> Result<DatasetData> {
        use ndarray_npy::{read_npy, NpzReader};
        
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
            
        if extension == "npy" {
            let array: ArrayD<f32> = read_npy(path)
                .map_err(|e| PhynexusError::IoError(format!("Failed to read NPY file: {}", e)))?;
                
            Ok(DatasetData::Array(array))
        } else if extension == "npz" {
            let mut npz = NpzReader::new(File::open(path)
                .map_err(|e| PhynexusError::IoError(format!("Failed to open NPZ file: {}", e)))?)?;
                
            let array_names = npz.array_names();
            if let Some(name) = array_names.first() {
                let array: ArrayD<f32> = npz.by_name(name)
                    .map_err(|e| PhynexusError::IoError(format!("Failed to read array '{}' from NPZ file: {}", name, e)))?;
                    
                Ok(DatasetData::Array(array))
            } else {
                Err(PhynexusError::IoError("NPZ file contains no arrays".to_string()))
            }
        } else {
            Err(PhynexusError::InvalidArgument(format!("Unsupported NumPy file extension: {}", extension)))
        }
    }
    
    fn load_text(&self, path: &Path, options: &HashMap<String, Value>) -> Result<DatasetData> {
        let encoding = options.get("encoding").and_then(|v| v.as_str()).unwrap_or("utf-8");
        
        let mut file = File::open(path)
            .map_err(|e| PhynexusError::IoError(format!("Failed to open file: {}", e)))?;
            
        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| PhynexusError::IoError(format!("Failed to read text file with encoding {}: {}", encoding, e)))?;
            
        Ok(DatasetData::Text(content))
    }
    
    fn load_image(&self, path: &Path, _options: &HashMap<String, Value>) -> Result<DatasetData> {
        let mut file = File::open(path)
            .map_err(|e| PhynexusError::IoError(format!("Failed to open file: {}", e)))?;
            
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| PhynexusError::IoError(format!("Failed to read image file: {}", e)))?;
            
        Ok(DatasetData::Bytes(bytes))
    }
    
    fn load_audio(&self, path: &Path, _options: &HashMap<String, Value>) -> Result<DatasetData> {
        let mut file = File::open(path)
            .map_err(|e| PhynexusError::IoError(format!("Failed to open file: {}", e)))?;
            
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| PhynexusError::IoError(format!("Failed to read audio file: {}", e)))?;
            
        Ok(DatasetData::Bytes(bytes))
    }
    
    fn load_video(&self, path: &Path, _options: &HashMap<String, Value>) -> Result<DatasetData> {
        let mut file = File::open(path)
            .map_err(|e| PhynexusError::IoError(format!("Failed to open file: {}", e)))?;
            
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| PhynexusError::IoError(format!("Failed to read video file: {}", e)))?;
            
        Ok(DatasetData::Bytes(bytes))
    }
}

#[pymodule]
fn dataset_hub(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_load_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(py_register_dataset, m)?)?;
    
    Ok(())
}

static mut DATASET_HUB: Option<Mutex<DatasetHub>> = None;

fn get_dataset_hub() -> &'static Mutex<DatasetHub> {
    unsafe {
        if DATASET_HUB.is_none() {
            DATASET_HUB = Some(Mutex::new(DatasetHub::new(None)));
        }
        DATASET_HUB.as_ref().unwrap()
    }
}

#[pyfunction]
fn py_load_dataset(
    py: Python,
    source: &str,
    format: Option<&str>,
    force_download: Option<bool>,
    options: Option<&PyDict>,
) -> PyResult<PyObject> {
    let format = if let Some(format_str) = format {
        Some(DatasetFormat::from_str(format_str)
            .map_err(|e| PyValueError::new_err(e.to_string()))?)
    } else {
        None
    };
    
    let options_json = if let Some(options_dict) = options {
        let mut map = HashMap::new();
        for (key, value) in options_dict.iter() {
            let key = key.extract::<String>()?;
            let value = if let Ok(s) = value.extract::<String>() {
                Value::String(s)
            } else if let Ok(i) = value.extract::<i64>() {
                Value::Number(serde_json::Number::from(i))
            } else if let Ok(f) = value.extract::<f64>() {
                if let Some(n) = serde_json::Number::from_f64(f) {
                    Value::Number(n)
                } else {
                    Value::Null
                }
            } else if let Ok(b) = value.extract::<bool>() {
                Value::Bool(b)
            } else {
                Value::Null
            };
            map.insert(key, value);
        }
        serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string())
    } else {
        "{}".to_string()
    };
    
    let dataset_hub = get_dataset_hub();
    let dataset = dataset_hub.lock().unwrap()
        .load_dataset(source, format, force_download.unwrap_or(false), &options_json)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    let data = match dataset.data.lock().unwrap().deref() {
        DatasetData::Tensor(tensor) => {
            let array = tensor.to_array()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            let np = py.import("numpy")?;
            let py_array = np.call_method1("array", (array.to_pyobject(py),))?;
            py_array.to_object(py)
        },
        DatasetData::Array(array) => {
            let np = py.import("numpy")?;
            let py_array = np.call_method1("array", (array.to_pyobject(py),))?;
            py_array.to_object(py)
        },
        DatasetData::Text(text) => {
            text.to_object(py)
        },
        DatasetData::Bytes(bytes) => {
            PyBytes::new(py, bytes).to_object(py)
        },
        DatasetData::Json(json) => {
            let json_str = json.to_string();
            let json_module = py.import("json")?;
            let py_obj = json_module.call_method1("loads", (json_str,))?;
            py_obj.to_object(py)
        },
        DatasetData::Table(table) => {
            let py_table = PyList::empty(py);
            for row in table {
                let py_row = PyList::empty(py);
                for cell in row {
                    py_row.append(PyString::new(py, cell))?;
                }
                py_table.append(py_row)?;
            }
            py_table.to_object(py)
        },
    };
    
    let dataset_module = py.import("neurenix.data.dataset_hub")?;
    let dataset_class = dataset_module.getattr("Dataset")?;
    
    let metadata = PyDict::new(py);
    for (key, value) in dataset.metadata() {
        metadata.set_item(key, value)?;
    }
    
    let format_enum = dataset_module.getattr("DatasetFormat")?;
    let format_value = match dataset.format() {
        DatasetFormat::CSV => format_enum.getattr("CSV")?,
        DatasetFormat::JSON => format_enum.getattr("JSON")?,
        DatasetFormat::NUMPY => format_enum.getattr("NUMPY")?,
        DatasetFormat::PICKLE => format_enum.getattr("PICKLE")?,
        DatasetFormat::TEXT => format_enum.getattr("TEXT")?,
        DatasetFormat::IMAGE => format_enum.getattr("IMAGE")?,
        DatasetFormat::AUDIO => format_enum.getattr("AUDIO")?,
        DatasetFormat::VIDEO => format_enum.getattr("VIDEO")?,
        DatasetFormat::CUSTOM => format_enum.getattr("CUSTOM")?,
    };
    
    let py_dataset = dataset_class.call1((
        data,
        format_value,
        dataset.name(),
        metadata,
    ))?;
    
    Ok(py_dataset.to_object(py))
}

#[pyfunction]
fn py_register_dataset(
    source: &str,
    url: &str,
    format: Option<&str>,
    metadata: Option<&PyDict>,
) -> PyResult<()> {
    let format = if let Some(format_str) = format {
        Some(DatasetFormat::from_str(format_str)
            .map_err(|e| PyValueError::new_err(e.to_string()))?)
    } else {
        None
    };
    
    let metadata_map = if let Some(metadata_dict) = metadata {
        let mut map = HashMap::new();
        for (key, value) in metadata_dict.iter() {
            let key = key.extract::<String>()?;
            let value = value.extract::<String>()?;
            map.insert(key, value);
        }
        Some(map)
    } else {
        None
    };
    
    let dataset_hub = get_dataset_hub();
    let mut hub = dataset_hub.lock().unwrap();
    hub.register_dataset(source, url, format, metadata_map);
    
    Ok(())
}

pub fn register_dataset_hub(py: Python, m: &PyModule) -> PyResult<()> {
    let dataset_hub_module = PyModule::new(py, "dataset_hub")?;
    
    dataset_hub(py, dataset_hub_module)?;
    
    m.add_submodule(dataset_hub_module)?;
    
    Ok(())
}
