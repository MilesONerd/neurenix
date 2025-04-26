"""
Unsupervised learning module for the Neurenix framework.

This module provides tools and utilities for unsupervised learning, allowing models
to learn patterns and representations from unlabeled data.
"""

from neurenix.unsupervised.autoencoder import Autoencoder, VAE, DenoisingAutoencoder
from neurenix.unsupervised.clustering import KMeans, DBSCAN, SpectralClustering
from neurenix.unsupervised.dim_reduction import PCA, TSNE, UMAP
from neurenix.unsupervised.contrastive import SimCLR, BYOL, MoCo

__all__ = [
    'Autoencoder',
    'VAE',
    'DenoisingAutoencoder',
    'KMeans',
    'DBSCAN',
    'SpectralClustering',
    'PCA',
    'TSNE',
    'UMAP',
    'SimCLR',
    'BYOL',
    'MoCo',
]
