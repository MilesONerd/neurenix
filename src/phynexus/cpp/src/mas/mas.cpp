/**
 * @file mas.cpp
 * @brief Multi-Agent Systems (MAS) implementation for Neurenix
 */

#include "mas/mas.h"
#include <algorithm>
#include <random>
#include <cmath>

namespace phynexus {
namespace mas {

Agent::Agent(const std::string& id) : id_(id) {}

std::string Agent::getId() const {
    return id_;
}

ReactiveAgent::ReactiveAgent(const std::string& id) : Agent(id) {}

void ReactiveAgent::perceive(const std::unordered_map<std::string, double>& observations) {
    state_ = observations;
}

std::string ReactiveAgent::selectAction() {
    for (const auto& rule : rules_) {
        if (rule.condition(state_)) {
            return rule.action;
        }
    }
    return "no_action";
}

void ReactiveAgent::update(double reward) {
}

void ReactiveAgent::addRule(std::function<bool(const std::unordered_map<std::string, double>&)> condition,
                           const std::string& action) {
    rules_.push_back({condition, action});
}

DeliberativeAgent::DeliberativeAgent(const std::string& id) : Agent(id) {}

void DeliberativeAgent::perceive(const std::unordered_map<std::string, double>& observations) {
    state_ = observations;
}

std::string DeliberativeAgent::selectAction() {
    if (actionModels_.empty()) {
        return "no_action";
    }

    std::string bestAction;
    double bestValue = -std::numeric_limits<double>::infinity();

    for (const auto& [action, model] : actionModels_) {
        auto resultState = model(state_);
        
        double value = 0.0;
        for (const auto& [key, goalValue] : goal_) {
            if (resultState.find(key) != resultState.end()) {
                double diff = goalValue - resultState[key];
                value -= diff * diff;  // Negative squared distance (higher is better)
            }
        }
        
        if (value > bestValue) {
            bestValue = value;
            bestAction = action;
        }
    }
    
    return bestAction;
}

void DeliberativeAgent::update(double reward) {
}

void DeliberativeAgent::setGoal(const std::unordered_map<std::string, double>& goal) {
    goal_ = goal;
}

void DeliberativeAgent::addActionModel(
    const std::string& action,
    std::function<std::unordered_map<std::string, double>(
        const std::unordered_map<std::string, double>&)> effects) {
    actionModels_[action] = effects;
}

Environment::Environment() {}

void Environment::addAgent(std::shared_ptr<Agent> agent) {
    agents_[agent->getId()] = agent;
}

void Environment::step() {
    for (auto& [agentId, agent] : agents_) {
        auto observations = getObservations(agentId);
        agent->perceive(observations);
        
        std::string action = agent->selectAction();
        double reward = processAction(agentId, action);
        
        agent->update(reward);
    }
}

CommunicationChannel::CommunicationChannel() {}

void CommunicationChannel::sendMessage(const std::string& senderId, const std::string& receiverId,
                                      const std::string& message) {
    messages_.push_back({senderId, receiverId, message});
}

std::vector<std::pair<std::string, std::string>> CommunicationChannel::getMessages(const std::string& agentId) {
    std::vector<std::pair<std::string, std::string>> result;
    
    for (const auto& message : messages_) {
        if (message.receiverId == agentId) {
            result.emplace_back(message.senderId, message.content);
        }
    }
    
    return result;
}

void CommunicationChannel::clearMessages() {
    messages_.clear();
}

CoordinationMechanism::CoordinationMechanism() {}

void CoordinationMechanism::registerAgent(const std::string& agentId) {
    registeredAgents_.push_back(agentId);
}

TaskAllocation::TaskAllocation() {}

void TaskAllocation::addTask(const std::string& taskId,
                            const std::unordered_map<std::string, double>& requirements) {
    tasks_[taskId] = requirements;
}

void TaskAllocation::setAgentCapabilities(const std::string& agentId,
                                         const std::unordered_map<std::string, double>& capabilities) {
    agentCapabilities_[agentId] = capabilities;
}

std::unordered_map<std::string, std::string> TaskAllocation::coordinate(
    const std::unordered_map<std::string, std::string>& agentActions) {
    
    std::unordered_map<std::string, std::string> result;
    std::unordered_map<std::string, bool> taskAssigned;
    
    for (const auto& agentId : registeredAgents_) {
        if (agentCapabilities_.find(agentId) == agentCapabilities_.end()) {
            result[agentId] = "no_task";
            continue;
        }
        
        const auto& capabilities = agentCapabilities_[agentId];
        
        std::string bestTask;
        double bestScore = -1.0;
        
        for (const auto& [taskId, requirements] : tasks_) {
            if (taskAssigned[taskId]) {
                continue;
            }
            
            double score = 0.0;
            int matchedRequirements = 0;
            
            for (const auto& [req, value] : requirements) {
                if (capabilities.find(req) != capabilities.end()) {
                    score += capabilities.at(req) / value;
                    matchedRequirements++;
                }
            }
            
            if (matchedRequirements == requirements.size() && score > bestScore) {
                bestScore = score;
                bestTask = taskId;
            }
        }
        
        if (!bestTask.empty()) {
            result[agentId] = bestTask;
            taskAssigned[bestTask] = true;
        } else {
            result[agentId] = "no_task";
        }
    }
    
    return result;
}

MultiAgentLearning::MultiAgentLearning() {}

IndependentQLearning::IndependentQLearning(double learningRate, double discountFactor)
    : learningRate_(learningRate), discountFactor_(discountFactor) {}

void IndependentQLearning::initialize(int numAgents, int stateSize, int actionSize) {
    qTables_.resize(numAgents);
}

void IndependentQLearning::update(int agentId, const std::vector<double>& state, int action, double reward,
                                 const std::vector<double>& nextState, bool done) {
    std::string stateStr = stateToString(state);
    std::string nextStateStr = stateToString(nextState);
    
    if (qTables_[agentId].find(stateStr) == qTables_[agentId].end()) {
        qTables_[agentId][stateStr] = std::vector<double>(getBestAction(agentId, nextState), 0.0);
    }
    
    if (qTables_[agentId].find(nextStateStr) == qTables_[agentId].end()) {
        qTables_[agentId][nextStateStr] = std::vector<double>(getBestAction(agentId, nextState), 0.0);
    }
    
    double maxNextQ = 0.0;
    if (!done) {
        maxNextQ = *std::max_element(qTables_[agentId][nextStateStr].begin(), 
                                     qTables_[agentId][nextStateStr].end());
    }
    
    double target = reward + discountFactor_ * maxNextQ;
    double currentQ = qTables_[agentId][stateStr][action];
    
    qTables_[agentId][stateStr][action] = currentQ + learningRate_ * (target - currentQ);
}

int IndependentQLearning::getBestAction(int agentId, const std::vector<double>& state) {
    std::string stateStr = stateToString(state);
    
    if (qTables_[agentId].find(stateStr) == qTables_[agentId].end()) {
        return rand() % qTables_[agentId].begin()->second.size();
    }
    
    const auto& qValues = qTables_[agentId][stateStr];
    return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
}

int IndependentQLearning::getExploratoryAction(int agentId, const std::vector<double>& state,
                                              double epsilon) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    if (dis(gen) < epsilon) {
        std::string stateStr = stateToString(state);
        int actionSize = qTables_[agentId].empty() ? 
                         0 : qTables_[agentId].begin()->second.size();
        
        if (actionSize == 0) {
            return 0;  // Default action if no information available
        }
        
        std::uniform_int_distribution<> actionDis(0, actionSize - 1);
        return actionDis(gen);
    } else {
        return getBestAction(agentId, state);
    }
}

std::string IndependentQLearning::stateToString(const std::vector<double>& state) {
    std::string result;
    for (double value : state) {
        result += std::to_string(value) + ",";
    }
    return result;
}

} // namespace mas
} // namespace phynexus
