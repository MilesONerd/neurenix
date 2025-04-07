"""
Coordination module for Multi-Agent Systems in Neurenix.

This module provides implementations of coordination mechanisms for
multi-agent systems, including task allocation, auctions, contract nets,
and voting.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np

class TaskAllocation:
    """Base class for task allocation mechanisms."""
    
    def __init__(self, name: str):
        """
        Initialize a task allocation mechanism.
        
        Args:
            name: Name of the mechanism
        """
        self.name = name
        self.tasks = {}
        self.agents = {}
        
    def add_task(self, task_id: str, requirements: Dict[str, Any]) -> None:
        """
        Add a task to be allocated.
        
        Args:
            task_id: Unique identifier for the task
            requirements: Requirements for the task
        """
        self.tasks[task_id] = {
            'id': task_id,
            'requirements': requirements,
            'assigned_to': None,
            'status': 'pending'
        }
        
    def add_agent(self, agent_id: str, capabilities: Dict[str, Any]) -> None:
        """
        Add an agent to the allocation mechanism.
        
        Args:
            agent_id: Unique identifier for the agent
            capabilities: Capabilities of the agent
        """
        self.agents[agent_id] = {
            'id': agent_id,
            'capabilities': capabilities,
            'assigned_tasks': []
        }
        
    def allocate(self) -> Dict[str, str]:
        """
        Allocate tasks to agents.
        
        Returns:
            Dictionary mapping task IDs to agent IDs
        """
        raise NotImplementedError("Subclasses must implement allocate()")


class Auction(TaskAllocation):
    """Task allocation through auction mechanisms."""
    
    def __init__(self, name: str, auction_type: str = "first-price"):
        """
        Initialize an auction-based task allocation mechanism.
        
        Args:
            name: Name of the mechanism
            auction_type: Type of auction (first-price, second-price, etc.)
        """
        super().__init__(name)
        self.auction_type = auction_type
        self.bids = {}
        
    def submit_bid(self, agent_id: str, task_id: str, bid: float) -> None:
        """
        Submit a bid for a task.
        
        Args:
            agent_id: ID of the bidding agent
            task_id: ID of the task being bid on
            bid: Bid value
        """
        if task_id not in self.bids:
            self.bids[task_id] = []
            
        self.bids[task_id].append({
            'agent_id': agent_id,
            'bid': bid
        })
        
    def allocate(self) -> Dict[str, str]:
        """
        Allocate tasks based on auction results.
        
        Returns:
            Dictionary mapping task IDs to agent IDs
        """
        allocations = {}
        
        for task_id, bids in self.bids.items():
            if not bids:
                continue
                
            sorted_bids = sorted(bids, key=lambda x: x['bid'])
            
            winner = sorted_bids[0]['agent_id']
            allocations[task_id] = winner
            
            if task_id in self.tasks:
                self.tasks[task_id]['assigned_to'] = winner
                self.tasks[task_id]['status'] = 'assigned'
                
            if winner in self.agents:
                self.agents[winner]['assigned_tasks'].append(task_id)
                
        return allocations


class ContractNet(TaskAllocation):
    """Task allocation through the Contract Net Protocol."""
    
    def __init__(self, name: str):
        """
        Initialize a Contract Net Protocol task allocation mechanism.
        
        Args:
            name: Name of the mechanism
        """
        super().__init__(name)
        self.proposals = {}
        self.awards = {}
        
    def announce_task(self, task_id: str) -> List[str]:
        """
        Announce a task to eligible agents.
        
        Args:
            task_id: ID of the task to announce
            
        Returns:
            List of agent IDs that received the announcement
        """
        if task_id not in self.tasks:
            return []
            
        eligible_agents = list(self.agents.keys())
        
        return eligible_agents
        
    def submit_proposal(self, agent_id: str, task_id: str, proposal: Dict[str, Any]) -> None:
        """
        Submit a proposal for a task.
        
        Args:
            agent_id: ID of the proposing agent
            task_id: ID of the task
            proposal: Proposal details
        """
        if task_id not in self.proposals:
            self.proposals[task_id] = []
            
        self.proposals[task_id].append({
            'agent_id': agent_id,
            'proposal': proposal
        })
        
    def award_task(self, task_id: str, agent_id: str) -> bool:
        """
        Award a task to an agent.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent
            
        Returns:
            Whether the award was successful
        """
        if task_id not in self.tasks or agent_id not in self.agents:
            return False
            
        self.awards[task_id] = agent_id
        self.tasks[task_id]['assigned_to'] = agent_id
        self.tasks[task_id]['status'] = 'assigned'
        self.agents[agent_id]['assigned_tasks'].append(task_id)
        
        return True
        
    def allocate(self) -> Dict[str, str]:
        """
        Allocate tasks using the Contract Net Protocol.
        
        Returns:
            Dictionary mapping task IDs to agent IDs
        """
        allocations = {}
        
        for task_id, proposals in self.proposals.items():
            if not proposals:
                continue
                
            best_proposal = proposals[0]
            agent_id = best_proposal['agent_id']
            
            if self.award_task(task_id, agent_id):
                allocations[task_id] = agent_id
                
        return allocations


class Voting:
    """Voting mechanisms for multi-agent decision making."""
    
    def __init__(self, name: str, voting_method: str = "majority"):
        """
        Initialize a voting mechanism.
        
        Args:
            name: Name of the mechanism
            voting_method: Voting method to use
        """
        self.name = name
        self.voting_method = voting_method
        self.voters = set()
        self.options = {}
        self.votes = {}
        
    def add_voter(self, voter_id: str) -> None:
        """
        Add a voter to the mechanism.
        
        Args:
            voter_id: ID of the voter
        """
        self.voters.add(voter_id)
        
    def add_option(self, option_id: str, description: str) -> None:
        """
        Add an option to vote on.
        
        Args:
            option_id: ID of the option
            description: Description of the option
        """
        self.options[option_id] = description
        
    def cast_vote(self, voter_id: str, option_id: str, weight: float = 1.0) -> bool:
        """
        Cast a vote for an option.
        
        Args:
            voter_id: ID of the voter
            option_id: ID of the option
            weight: Weight of the vote
            
        Returns:
            Whether the vote was successfully cast
        """
        if voter_id not in self.voters or option_id not in self.options:
            return False
            
        if voter_id not in self.votes:
            self.votes[voter_id] = {}
            
        self.votes[voter_id][option_id] = weight
        
        return True
        
    def tally_votes(self) -> Dict[str, float]:
        """
        Tally the votes.
        
        Returns:
            Dictionary mapping option IDs to vote counts/scores
        """
        tally = {option_id: 0.0 for option_id in self.options}
        
        for voter_id, votes in self.votes.items():
            for option_id, weight in votes.items():
                tally[option_id] += weight
                
        return tally
        
    def get_winner(self) -> str:
        """
        Get the winning option.
        
        Returns:
            ID of the winning option
        """
        tally = self.tally_votes()
        
        if not tally:
            return None
            
        if self.voting_method == "majority":
            return max(tally.items(), key=lambda x: x[1])[0]
        elif self.voting_method == "approval":
            approval_count = {option_id: 0 for option_id in self.options}
            for voter_id, votes in self.votes.items():
                for option_id in votes:
                    approval_count[option_id] += 1
            return max(approval_count.items(), key=lambda x: x[1])[0]
        else:
            return max(tally.items(), key=lambda x: x[1])[0]
