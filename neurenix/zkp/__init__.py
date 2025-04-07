"""
Zero-Knowledge Proofs (ZKP) module for Neurenix.

This module provides implementations of various zero-knowledge proof systems
that allow one party (the prover) to prove to another party (the verifier)
that a statement is true without revealing any additional information.
"""

from .proof_system import ProofSystem, Prover, Verifier
from .zk_snark import ZKSnark, ZKSnarkProver, ZKSnarkVerifier
from .zk_stark import ZKStark, ZKStarkProver, ZKStarkVerifier
from .bulletproofs import Bulletproofs, BulletproofsProver, BulletproofsVerifier
from .sigma import SigmaProtocol, SigmaProver, SigmaVerifier
from .utils import generate_parameters, verify_proof

__all__ = [
    'ProofSystem', 'Prover', 'Verifier',
    'ZKSnark', 'ZKSnarkProver', 'ZKSnarkVerifier',
    'ZKStark', 'ZKStarkProver', 'ZKStarkVerifier',
    'Bulletproofs', 'BulletproofsProver', 'BulletproofsVerifier',
    'SigmaProtocol', 'SigmaProver', 'SigmaVerifier',
    'generate_parameters', 'verify_proof',
]
