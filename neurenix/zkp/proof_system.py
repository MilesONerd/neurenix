"""
Base classes for zero-knowledge proof systems in Neurenix.

This module provides the abstract base classes for proof systems,
provers, and verifiers that are used throughout the ZKP module.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from neurenix.tensor import Tensor


class ProofSystem(ABC):
    """Base class for all zero-knowledge proof systems."""
    
    def __init__(self, security_parameter: int = 128):
        """Initialize a proof system with the given security parameter.
        
        Args:
            security_parameter: The security parameter in bits (default: 128).
        """
        self.security_parameter = security_parameter
        self.setup_params = None
    
    @abstractmethod
    def setup(self, **kwargs) -> Dict[str, Any]:
        """Set up the proof system with the necessary parameters.
        
        Returns:
            A dictionary containing the setup parameters.
        """
        pass
    
    def get_prover(self) -> 'Prover':
        """Get a prover for this proof system.
        
        Returns:
            A prover instance for this proof system.
        """
        if self.setup_params is None:
            self.setup_params = self.setup()
        return self._create_prover(self.setup_params)
    
    def get_verifier(self) -> 'Verifier':
        """Get a verifier for this proof system.
        
        Returns:
            A verifier instance for this proof system.
        """
        if self.setup_params is None:
            self.setup_params = self.setup()
        return self._create_verifier(self.setup_params)
    
    @abstractmethod
    def _create_prover(self, setup_params: Dict[str, Any]) -> 'Prover':
        """Create a prover for this proof system.
        
        Args:
            setup_params: The setup parameters for the proof system.
            
        Returns:
            A prover instance for this proof system.
        """
        pass
    
    @abstractmethod
    def _create_verifier(self, setup_params: Dict[str, Any]) -> 'Verifier':
        """Create a verifier for this proof system.
        
        Args:
            setup_params: The setup parameters for the proof system.
            
        Returns:
            A verifier instance for this proof system.
        """
        pass


class Prover(ABC):
    """Base class for all zero-knowledge provers."""
    
    def __init__(self, setup_params: Dict[str, Any]):
        """Initialize a prover with the given setup parameters.
        
        Args:
            setup_params: The setup parameters for the proof system.
        """
        self.setup_params = setup_params
    
    @abstractmethod
    def prove(self, statement: Any, witness: Any, **kwargs) -> Dict[str, Any]:
        """Generate a proof for the given statement and witness.
        
        Args:
            statement: The statement to prove.
            witness: The witness for the statement.
            **kwargs: Additional arguments for the proof generation.
            
        Returns:
            A dictionary containing the proof.
        """
        pass


class Verifier(ABC):
    """Base class for all zero-knowledge verifiers."""
    
    def __init__(self, setup_params: Dict[str, Any]):
        """Initialize a verifier with the given setup parameters.
        
        Args:
            setup_params: The setup parameters for the proof system.
        """
        self.setup_params = setup_params
    
    @abstractmethod
    def verify(self, statement: Any, proof: Dict[str, Any], **kwargs) -> bool:
        """Verify a proof for the given statement.
        
        Args:
            statement: The statement to verify.
            proof: The proof to verify.
            **kwargs: Additional arguments for the verification.
            
        Returns:
            True if the proof is valid, False otherwise.
        """
        pass


class InteractiveProofSystem(ProofSystem):
    """Base class for interactive zero-knowledge proof systems."""
    
    @abstractmethod
    def generate_challenge(self, commitment: Any) -> Any:
        """Generate a challenge for the prover based on their commitment.
        
        Args:
            commitment: The prover's commitment.
            
        Returns:
            A challenge for the prover.
        """
        pass


class InteractiveProver(Prover):
    """Base class for interactive zero-knowledge provers."""
    
    @abstractmethod
    def commit(self, statement: Any, witness: Any) -> Any:
        """Generate a commitment for the given statement and witness.
        
        Args:
            statement: The statement to prove.
            witness: The witness for the statement.
            
        Returns:
            A commitment.
        """
        pass
    
    @abstractmethod
    def respond(self, statement: Any, witness: Any, commitment: Any, challenge: Any) -> Any:
        """Generate a response to the given challenge.
        
        Args:
            statement: The statement to prove.
            witness: The witness for the statement.
            commitment: The prover's commitment.
            challenge: The verifier's challenge.
            
        Returns:
            A response to the challenge.
        """
        pass
    
    def prove(self, statement: Any, witness: Any, verifier: 'InteractiveVerifier', **kwargs) -> Dict[str, Any]:
        """Generate a proof for the given statement and witness.
        
        Args:
            statement: The statement to prove.
            witness: The witness for the statement.
            verifier: The verifier to interact with.
            **kwargs: Additional arguments for the proof generation.
            
        Returns:
            A dictionary containing the proof transcript.
        """
        commitment = self.commit(statement, witness)
        challenge = verifier.challenge(commitment)
        response = self.respond(statement, witness, commitment, challenge)
        
        return {
            'commitment': commitment,
            'challenge': challenge,
            'response': response
        }


class InteractiveVerifier(Verifier):
    """Base class for interactive zero-knowledge verifiers."""
    
    @abstractmethod
    def challenge(self, commitment: Any) -> Any:
        """Generate a challenge for the prover based on their commitment.
        
        Args:
            commitment: The prover's commitment.
            
        Returns:
            A challenge for the prover.
        """
        pass
    
    @abstractmethod
    def check(self, statement: Any, commitment: Any, challenge: Any, response: Any) -> bool:
        """Check if the prover's response is valid.
        
        Args:
            statement: The statement to verify.
            commitment: The prover's commitment.
            challenge: The verifier's challenge.
            response: The prover's response.
            
        Returns:
            True if the response is valid, False otherwise.
        """
        pass
    
    def verify(self, statement: Any, proof: Dict[str, Any], **kwargs) -> bool:
        """Verify a proof for the given statement.
        
        Args:
            statement: The statement to verify.
            proof: The proof to verify.
            **kwargs: Additional arguments for the verification.
            
        Returns:
            True if the proof is valid, False otherwise.
        """
        commitment = proof.get('commitment')
        challenge = proof.get('challenge')
        response = proof.get('response')
        
        if commitment is None or challenge is None or response is None:
            return False
        
        return self.check(statement, commitment, challenge, response)


class NonInteractiveProofSystem(ProofSystem):
    """Base class for non-interactive zero-knowledge proof systems."""
    
    @abstractmethod
    def generate_crs(self, **kwargs) -> Dict[str, Any]:
        """Generate a common reference string for the proof system.
        
        Returns:
            A dictionary containing the common reference string.
        """
        pass
    
    def setup(self, **kwargs) -> Dict[str, Any]:
        """Set up the proof system with the necessary parameters.
        
        Returns:
            A dictionary containing the setup parameters.
        """
        return {'crs': self.generate_crs(**kwargs)}


class NonInteractiveProver(Prover):
    """Base class for non-interactive zero-knowledge provers."""
    
    @property
    def crs(self) -> Dict[str, Any]:
        """Get the common reference string for the proof system.
        
        Returns:
            The common reference string.
        """
        return self.setup_params.get('crs', {})


class NonInteractiveVerifier(Verifier):
    """Base class for non-interactive zero-knowledge verifiers."""
    
    @property
    def crs(self) -> Dict[str, Any]:
        """Get the common reference string for the proof system.
        
        Returns:
            The common reference string.
        """
        return self.setup_params.get('crs', {})
