"""
Defuzzification methods for Neurenix.

This module provides implementations of various defuzzification methods
for fuzzy logic systems.
"""

from typing import Optional, Union, Tuple

import neurenix as nx


def centroid(universe: nx.Tensor, membership: nx.Tensor) -> float:
    """
    Centroid defuzzification method.
    
    Args:
        universe: Universe of discourse
        membership: Membership degrees
        
    Returns:
        Defuzzified value
    """
    if universe.shape != membership.shape:
        raise ValueError("Universe and membership must have the same shape")
    
    if nx.sum(membership) == 0:
        return 0.0
    
    return nx.sum(universe * membership) / nx.sum(membership)


def bisector(universe: nx.Tensor, membership: nx.Tensor) -> float:
    """
    Bisector defuzzification method.
    
    Args:
        universe: Universe of discourse
        membership: Membership degrees
        
    Returns:
        Defuzzified value
    """
    if universe.shape != membership.shape:
        raise ValueError("Universe and membership must have the same shape")
    
    if nx.sum(membership) == 0:
        return 0.0
    
    cumsum = nx.cumsum(membership)
    
    half_sum = cumsum[-1] / 2
    
    idx = nx.argmin(nx.abs(cumsum - half_sum))
    
    return universe[idx].item()


def mean_of_maximum(universe: nx.Tensor, membership: nx.Tensor) -> float:
    """
    Mean of maximum defuzzification method.
    
    Args:
        universe: Universe of discourse
        membership: Membership degrees
        
    Returns:
        Defuzzified value
    """
    if universe.shape != membership.shape:
        raise ValueError("Universe and membership must have the same shape")
    
    max_membership = nx.max(membership)
    
    if max_membership == 0:
        return 0.0
    
    max_indices = nx.where(membership == max_membership)[0]
    
    return nx.mean(universe[max_indices]).item()


def smallest_of_maximum(universe: nx.Tensor, membership: nx.Tensor) -> float:
    """
    Smallest of maximum defuzzification method.
    
    Args:
        universe: Universe of discourse
        membership: Membership degrees
        
    Returns:
        Defuzzified value
    """
    if universe.shape != membership.shape:
        raise ValueError("Universe and membership must have the same shape")
    
    max_membership = nx.max(membership)
    
    if max_membership == 0:
        return 0.0
    
    max_indices = nx.where(membership == max_membership)[0]
    
    return universe[max_indices[0]].item()


def largest_of_maximum(universe: nx.Tensor, membership: nx.Tensor) -> float:
    """
    Largest of maximum defuzzification method.
    
    Args:
        universe: Universe of discourse
        membership: Membership degrees
        
    Returns:
        Defuzzified value
    """
    if universe.shape != membership.shape:
        raise ValueError("Universe and membership must have the same shape")
    
    max_membership = nx.max(membership)
    
    if max_membership == 0:
        return 0.0
    
    max_indices = nx.where(membership == max_membership)[0]
    
    return universe[max_indices[-1]].item()


def weighted_average(universe: nx.Tensor, membership: nx.Tensor) -> float:
    """
    Weighted average defuzzification method.
    
    Args:
        universe: Universe of discourse
        membership: Membership degrees
        
    Returns:
        Defuzzified value
    """
    if universe.shape != membership.shape:
        raise ValueError("Universe and membership must have the same shape")
    
    if nx.sum(membership) == 0:
        return 0.0
    
    return nx.sum(universe * membership) / nx.sum(membership)


def weighted_sum(universe: nx.Tensor, membership: nx.Tensor) -> float:
    """
    Weighted sum defuzzification method.
    
    Args:
        universe: Universe of discourse
        membership: Membership degrees
        
    Returns:
        Defuzzified value
    """
    if universe.shape != membership.shape:
        raise ValueError("Universe and membership must have the same shape")
    
    return nx.sum(universe * membership).item()


def quality_method(universe: nx.Tensor, membership: nx.Tensor, alpha: float = 0.5) -> float:
    """
    Quality method defuzzification.
    
    Args:
        universe: Universe of discourse
        membership: Membership degrees
        alpha: Quality parameter (0 <= alpha <= 1)
        
    Returns:
        Defuzzified value
    """
    if universe.shape != membership.shape:
        raise ValueError("Universe and membership must have the same shape")
    
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    alpha_cut = nx.where(membership >= alpha, membership, nx.zeros_like(membership))
    
    if nx.sum(alpha_cut) == 0:
        return 0.0
    
    return nx.sum(universe * alpha_cut) / nx.sum(alpha_cut)


def extended_quality_method(universe: nx.Tensor, membership: nx.Tensor, 
                           alpha: float = 0.5, beta: float = 0.5) -> float:
    """
    Extended quality method defuzzification.
    
    Args:
        universe: Universe of discourse
        membership: Membership degrees
        alpha: Quality parameter (0 <= alpha <= 1)
        beta: Quality parameter (0 <= beta <= 1)
        
    Returns:
        Defuzzified value
    """
    if universe.shape != membership.shape:
        raise ValueError("Universe and membership must have the same shape")
    
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    if beta < 0 or beta > 1:
        raise ValueError("Beta must be between 0 and 1")
    
    alpha_cut = nx.where(membership >= alpha, membership, nx.zeros_like(membership))
    
    beta_cut = nx.where(membership >= beta, membership, nx.zeros_like(membership))
    
    if nx.sum(alpha_cut) == 0 or nx.sum(beta_cut) == 0:
        return 0.0
    
    centroid_alpha = nx.sum(universe * alpha_cut) / nx.sum(alpha_cut)
    
    centroid_beta = nx.sum(universe * beta_cut) / nx.sum(beta_cut)
    
    return (centroid_alpha + centroid_beta) / 2
