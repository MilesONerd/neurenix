//! Learning algorithms for the Phynexus engine

pub mod transfer;
pub mod meta;
pub mod unsupervised;

// Re-export common types
pub use transfer::TransferModel;
pub use meta::MetaLearningModel;
pub use unsupervised::Autoencoder;
