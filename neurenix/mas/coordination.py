"""
Coordination mechanisms for Multi-Agent Systems in Neurenix.

This module provides implementations of coordination mechanisms for multi-agent systems,
including auctions, contract nets, voting, and coalition formation.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from collections import defaultdict

from neurenix.tensor import Tensor
from .communication import Message


class Coordinator:
    """Base class for coordination mechanisms in multi-agent systems."""
    
    def __init__(self, coordinator_id: str):
        """Initialize a coordinator.
        
        Args:
            coordinator_id: Unique identifier for the coordinator
        """
        self.coordinator_id = coordinator_id
        self.agents = {}
        self.coordination_state = {}
        
    def register_agent(self, agent_id: str, agent_info: Optional[Dict[str, Any]] = None) -> None:
        """Register an agent with the coordinator.
        
        Args:
            agent_id: ID of the agent to register
            agent_info: Optional information about the agent
        """
        self.agents[agent_id] = agent_info or {}
        
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the coordinator.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            
    def coordinate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents based on the current state.
        
        Args:
            state: Current state of the multi-agent system
            
        Returns:
            Coordination decisions
        """
        raise NotImplementedError("Coordinators must implement coordinate method")
    
    def reset(self) -> None:
        """Reset the coordinator to its initial state."""
        self.coordination_state = {}


class Auction(Coordinator):
    """Auction-based coordination mechanism."""
    
    def __init__(self, coordinator_id: str, auction_type: str = 'first_price'):
        """Initialize an auction coordinator.
        
        Args:
            coordinator_id: Unique identifier for the coordinator
            auction_type: Type of auction ('first_price', 'second_price', 'english', 'dutch')
        """
        super().__init__(coordinator_id)
        self.auction_type = auction_type
        self.auctions = {}
        self.bids = defaultdict(dict)
        self.winners = {}
        
    def create_auction(self, auction_id: str, item: Any, 
                     reserve_price: float = 0.0,
                     duration: int = 1) -> None:
        """Create a new auction.
        
        Args:
            auction_id: Unique identifier for the auction
            item: Item being auctioned
            reserve_price: Minimum acceptable price
            duration: Number of rounds the auction will run
        """
        self.auctions[auction_id] = {
            'item': item,
            'reserve_price': reserve_price,
            'duration': duration,
            'current_round': 0,
            'status': 'open',
            'highest_bid': reserve_price,
            'highest_bidder': None
        }
        
    def place_bid(self, auction_id: str, agent_id: str, bid_amount: float) -> bool:
        """Place a bid in an auction.
        
        Args:
            auction_id: ID of the auction
            agent_id: ID of the bidding agent
            bid_amount: Amount of the bid
            
        Returns:
            True if bid was accepted, False otherwise
        """
        if auction_id not in self.auctions:
            return False
            
        auction = self.auctions[auction_id]
        
        if auction['status'] != 'open':
            return False
            
        if bid_amount < auction['reserve_price']:
            return False
            
        self.bids[auction_id][agent_id] = bid_amount
        
        if bid_amount > auction['highest_bid']:
            auction['highest_bid'] = bid_amount
            auction['highest_bidder'] = agent_id
            
        return True
    
    def advance_auction(self, auction_id: str) -> Dict[str, Any]:
        """Advance an auction to the next round or close it.
        
        Args:
            auction_id: ID of the auction
            
        Returns:
            Updated auction state
        """
        if auction_id not in self.auctions:
            return {}
            
        auction = self.auctions[auction_id]
        
        if auction['status'] != 'open':
            return auction
            
        auction['current_round'] += 1
        
        if auction['current_round'] >= auction['duration']:
            self._close_auction(auction_id)
            
        return auction
    
    def _close_auction(self, auction_id: str) -> None:
        """Close an auction and determine the winner.
        
        Args:
            auction_id: ID of the auction
        """
        auction = self.auctions[auction_id]
        auction['status'] = 'closed'
        
        if auction['highest_bidder'] is None:
            return
            
        if self.auction_type == 'first_price':
            self.winners[auction_id] = {
                'agent_id': auction['highest_bidder'],
                'price': auction['highest_bid']
            }
        elif self.auction_type == 'second_price':
            sorted_bids = sorted(self.bids[auction_id].values(), reverse=True)
            second_price = sorted_bids[1] if len(sorted_bids) > 1 else auction['reserve_price']
            
            self.winners[auction_id] = {
                'agent_id': auction['highest_bidder'],
                'price': second_price
            }
        
    def coordinate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents through auctions.
        
        Args:
            state: Current state of the multi-agent system
            
        Returns:
            Coordination decisions
        """
        decisions = {
            'auctions': self.auctions,
            'winners': self.winners
        }
        
        if 'auction_actions' in state:
            for action in state['auction_actions']:
                if action['type'] == 'create_auction':
                    self.create_auction(
                        action['auction_id'],
                        action['item'],
                        action.get('reserve_price', 0.0),
                        action.get('duration', 1)
                    )
                elif action['type'] == 'place_bid':
                    self.place_bid(
                        action['auction_id'],
                        action['agent_id'],
                        action['bid_amount']
                    )
                elif action['type'] == 'advance_auction':
                    self.advance_auction(action['auction_id'])
        
        return decisions
    
    def reset(self) -> None:
        """Reset the auction coordinator."""
        super().reset()
        self.auctions = {}
        self.bids = defaultdict(dict)
        self.winners = {}


class ContractNet(Coordinator):
    """Contract Net Protocol for task allocation."""
    
    def __init__(self, coordinator_id: str):
        """Initialize a Contract Net coordinator.
        
        Args:
            coordinator_id: Unique identifier for the coordinator
        """
        super().__init__(coordinator_id)
        self.tasks = {}
        self.proposals = defaultdict(dict)
        self.contracts = {}
        
    def announce_task(self, task_id: str, task_description: Dict[str, Any],
                    deadline: int = 1) -> None:
        """Announce a task to all registered agents.
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of the task
            deadline: Number of rounds until proposals are due
        """
        self.tasks[task_id] = {
            'description': task_description,
            'deadline': deadline,
            'current_round': 0,
            'status': 'announced',
            'manager': task_description.get('manager')
        }
        
    def submit_proposal(self, task_id: str, agent_id: str, 
                      proposal: Dict[str, Any]) -> bool:
        """Submit a proposal for a task.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the proposing agent
            proposal: Proposal details
            
        Returns:
            True if proposal was accepted, False otherwise
        """
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        if task['status'] != 'announced':
            return False
            
        self.proposals[task_id][agent_id] = proposal
        return True
    
    def award_contract(self, task_id: str, agent_id: str) -> bool:
        """Award a contract to an agent.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent to award the contract to
            
        Returns:
            True if contract was awarded, False otherwise
        """
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        if task['status'] != 'announced':
            return False
            
        if agent_id not in self.proposals[task_id]:
            return False
            
        task['status'] = 'awarded'
        self.contracts[task_id] = {
            'agent_id': agent_id,
            'proposal': self.proposals[task_id][agent_id]
        }
        
        return True
    
    def advance_task(self, task_id: str) -> Dict[str, Any]:
        """Advance a task to the next round or close it.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Updated task state
        """
        if task_id not in self.tasks:
            return {}
            
        task = self.tasks[task_id]
        
        if task['status'] != 'announced':
            return task
            
        task['current_round'] += 1
        
        if task['current_round'] >= task['deadline']:
            task['status'] = 'deadline_passed'
            
        return task
    
    def coordinate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents through the Contract Net Protocol.
        
        Args:
            state: Current state of the multi-agent system
            
        Returns:
            Coordination decisions
        """
        decisions = {
            'tasks': self.tasks,
            'proposals': dict(self.proposals),
            'contracts': self.contracts
        }
        
        if 'contract_net_actions' in state:
            for action in state['contract_net_actions']:
                if action['type'] == 'announce_task':
                    self.announce_task(
                        action['task_id'],
                        action['task_description'],
                        action.get('deadline', 1)
                    )
                elif action['type'] == 'submit_proposal':
                    self.submit_proposal(
                        action['task_id'],
                        action['agent_id'],
                        action['proposal']
                    )
                elif action['type'] == 'award_contract':
                    self.award_contract(
                        action['task_id'],
                        action['agent_id']
                    )
                elif action['type'] == 'advance_task':
                    self.advance_task(action['task_id'])
        
        return decisions
    
    def reset(self) -> None:
        """Reset the Contract Net coordinator."""
        super().reset()
        self.tasks = {}
        self.proposals = defaultdict(dict)
        self.contracts = {}


class VotingMechanism(Coordinator):
    """Voting-based coordination mechanism."""
    
    def __init__(self, coordinator_id: str, voting_rule: str = 'plurality'):
        """Initialize a voting coordinator.
        
        Args:
            coordinator_id: Unique identifier for the coordinator
            voting_rule: Voting rule to use ('plurality', 'borda', 'approval', 'runoff')
        """
        super().__init__(coordinator_id)
        self.voting_rule = voting_rule
        self.elections = {}
        self.votes = defaultdict(dict)
        self.results = {}
        
    def create_election(self, election_id: str, candidates: List[Any],
                      deadline: int = 1) -> None:
        """Create a new election.
        
        Args:
            election_id: Unique identifier for the election
            candidates: List of candidates
            deadline: Number of rounds until voting closes
        """
        self.elections[election_id] = {
            'candidates': candidates,
            'deadline': deadline,
            'current_round': 0,
            'status': 'open'
        }
        
    def cast_vote(self, election_id: str, agent_id: str, 
                vote: Union[Any, List[Any]]) -> bool:
        """Cast a vote in an election.
        
        Args:
            election_id: ID of the election
            agent_id: ID of the voting agent
            vote: Vote (single candidate or ranked list depending on voting rule)
            
        Returns:
            True if vote was accepted, False otherwise
        """
        if election_id not in self.elections:
            return False
            
        election = self.elections[election_id]
        
        if election['status'] != 'open':
            return False
            
        if self.voting_rule == 'plurality':
            if vote not in election['candidates']:
                return False
        elif self.voting_rule == 'borda' or self.voting_rule == 'runoff':
            if not isinstance(vote, list) or not all(c in election['candidates'] for c in vote):
                return False
        elif self.voting_rule == 'approval':
            if not isinstance(vote, list) or not all(c in election['candidates'] for c in vote):
                return False
        
        self.votes[election_id][agent_id] = vote
        return True
    
    def advance_election(self, election_id: str) -> Dict[str, Any]:
        """Advance an election to the next round or close it.
        
        Args:
            election_id: ID of the election
            
        Returns:
            Updated election state
        """
        if election_id not in self.elections:
            return {}
            
        election = self.elections[election_id]
        
        if election['status'] != 'open':
            return election
            
        election['current_round'] += 1
        
        if election['current_round'] >= election['deadline']:
            self._close_election(election_id)
            
        return election
    
    def _close_election(self, election_id: str) -> None:
        """Close an election and determine the winner.
        
        Args:
            election_id: ID of the election
        """
        election = self.elections[election_id]
        election['status'] = 'closed'
        
        if not self.votes[election_id]:
            self.results[election_id] = {
                'winner': None,
                'tallies': {}
            }
            return
            
        if self.voting_rule == 'plurality':
            tallies = defaultdict(int)
            for vote in self.votes[election_id].values():
                tallies[vote] += 1
                
            winner = max(tallies.items(), key=lambda x: x[1])[0]
            
            self.results[election_id] = {
                'winner': winner,
                'tallies': dict(tallies)
            }
            
        elif self.voting_rule == 'borda':
            tallies = defaultdict(int)
            num_candidates = len(election['candidates'])
            
            for vote in self.votes[election_id].values():
                for i, candidate in enumerate(vote):
                    tallies[candidate] += num_candidates - i
                    
            winner = max(tallies.items(), key=lambda x: x[1])[0]
            
            self.results[election_id] = {
                'winner': winner,
                'tallies': dict(tallies)
            }
            
        elif self.voting_rule == 'approval':
            tallies = defaultdict(int)
            
            for vote in self.votes[election_id].values():
                for candidate in vote:
                    tallies[candidate] += 1
                    
            winner = max(tallies.items(), key=lambda x: x[1])[0]
            
            self.results[election_id] = {
                'winner': winner,
                'tallies': dict(tallies)
            }
            
        elif self.voting_rule == 'runoff':
            remaining_candidates = set(election['candidates'])
            tallies = defaultdict(int)
            
            while remaining_candidates:
                tallies = defaultdict(int)
                
                for vote in self.votes[election_id].values():
                    for candidate in vote:
                        if candidate in remaining_candidates:
                            tallies[candidate] += 1
                            break
                
                if not tallies:
                    winner = None
                    break
                    
                total_votes = sum(tallies.values())
                for candidate, count in tallies.items():
                    if count > total_votes / 2:
                        winner = candidate
                        break
                else:
                    loser = min(tallies.items(), key=lambda x: x[1])[0]
                    remaining_candidates.remove(loser)
                    continue
                    
                break
            
            self.results[election_id] = {
                'winner': winner if 'winner' in locals() else None,
                'tallies': dict(tallies)
            }
    
    def coordinate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents through voting.
        
        Args:
            state: Current state of the multi-agent system
            
        Returns:
            Coordination decisions
        """
        decisions = {
            'elections': self.elections,
            'results': self.results
        }
        
        if 'voting_actions' in state:
            for action in state['voting_actions']:
                if action['type'] == 'create_election':
                    self.create_election(
                        action['election_id'],
                        action['candidates'],
                        action.get('deadline', 1)
                    )
                elif action['type'] == 'cast_vote':
                    self.cast_vote(
                        action['election_id'],
                        action['agent_id'],
                        action['vote']
                    )
                elif action['type'] == 'advance_election':
                    self.advance_election(action['election_id'])
        
        return decisions
    
    def reset(self) -> None:
        """Reset the voting coordinator."""
        super().reset()
        self.elections = {}
        self.votes = defaultdict(dict)
        self.results = {}


class CoalitionFormation(Coordinator):
    """Coalition formation mechanism for multi-agent systems."""
    
    def __init__(self, coordinator_id: str, formation_method: str = 'greedy'):
        """Initialize a coalition formation coordinator.
        
        Args:
            coordinator_id: Unique identifier for the coordinator
            formation_method: Method for forming coalitions ('greedy', 'optimal', 'stable')
        """
        super().__init__(coordinator_id)
        self.formation_method = formation_method
        self.coalitions = {}
        self.agent_values = {}
        self.agent_preferences = {}
        
    def set_agent_value(self, agent_id: str, value: float) -> None:
        """Set the value of an agent for coalition formation.
        
        Args:
            agent_id: ID of the agent
            value: Value of the agent
        """
        self.agent_values[agent_id] = value
        
    def set_agent_preferences(self, agent_id: str, 
                            preferences: Dict[str, float]) -> None:
        """Set the preferences of an agent for other agents.
        
        Args:
            agent_id: ID of the agent
            preferences: Dictionary mapping other agent IDs to preference values
        """
        self.agent_preferences[agent_id] = preferences
        
    def form_coalitions(self, num_coalitions: int) -> Dict[str, List[str]]:
        """Form coalitions among registered agents.
        
        Args:
            num_coalitions: Number of coalitions to form
            
        Returns:
            Dictionary mapping coalition IDs to lists of agent IDs
        """
        if self.formation_method == 'greedy':
            return self._form_coalitions_greedy(num_coalitions)
        elif self.formation_method == 'optimal':
            return self._form_coalitions_optimal(num_coalitions)
        elif self.formation_method == 'stable':
            return self._form_coalitions_stable(num_coalitions)
        else:
            return {}
    
    def _form_coalitions_greedy(self, num_coalitions: int) -> Dict[str, List[str]]:
        """Form coalitions using a greedy algorithm.
        
        Args:
            num_coalitions: Number of coalitions to form
            
        Returns:
            Dictionary mapping coalition IDs to lists of agent IDs
        """
        sorted_agents = sorted(
            self.agent_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        coalitions = {f"coalition_{i}": [] for i in range(num_coalitions)}
        
        for i, (agent_id, _) in enumerate(sorted_agents):
            coalition_id = f"coalition_{i % num_coalitions}"
            coalitions[coalition_id].append(agent_id)
            
        self.coalitions = coalitions
        return coalitions
    
    def _form_coalitions_optimal(self, num_coalitions: int) -> Dict[str, List[str]]:
        """Form coalitions optimally (simplified implementation).
        
        Args:
            num_coalitions: Number of coalitions to form
            
        Returns:
            Dictionary mapping coalition IDs to lists of agent IDs
        """
        return self._form_coalitions_greedy(num_coalitions)
    
    def _form_coalitions_stable(self, num_coalitions: int) -> Dict[str, List[str]]:
        """Form stable coalitions based on agent preferences.
        
        Args:
            num_coalitions: Number of coalitions to form
            
        Returns:
            Dictionary mapping coalition IDs to lists of agent IDs
        """
        return self._form_coalitions_greedy(num_coalitions)
    
    def get_coalition_value(self, coalition: List[str]) -> float:
        """Calculate the value of a coalition.
        
        Args:
            coalition: List of agent IDs in the coalition
            
        Returns:
            Value of the coalition
        """
        return sum(self.agent_values.get(agent_id, 0.0) for agent_id in coalition)
    
    def get_agent_coalition(self, agent_id: str) -> Optional[str]:
        """Get the coalition an agent belongs to.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            ID of the coalition the agent belongs to, or None
        """
        for coalition_id, agents in self.coalitions.items():
            if agent_id in agents:
                return coalition_id
        return None
    
    def coordinate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agents through coalition formation.
        
        Args:
            state: Current state of the multi-agent system
            
        Returns:
            Coordination decisions
        """
        decisions = {
            'coalitions': self.coalitions
        }
        
        if 'coalition_actions' in state:
            for action in state['coalition_actions']:
                if action['type'] == 'set_agent_value':
                    self.set_agent_value(
                        action['agent_id'],
                        action['value']
                    )
                elif action['type'] == 'set_agent_preferences':
                    self.set_agent_preferences(
                        action['agent_id'],
                        action['preferences']
                    )
                elif action['type'] == 'form_coalitions':
                    self.form_coalitions(action['num_coalitions'])
        
        return decisions
    
    def reset(self) -> None:
        """Reset the coalition formation coordinator."""
        super().reset()
        self.coalitions = {}
