/**
 * @file mas.h
 * @brief Multi-Agent Systems (MAS) header for Neurenix
 */

#ifndef PHYNEXUS_MAS_H
#define PHYNEXUS_MAS_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace phynexus {
namespace mas {

/**
 * @brief Base class for all agents in a multi-agent system
 */
class Agent {
public:
    Agent(const std::string& id);
    virtual ~Agent() = default;

    /**
     * @brief Get the agent's unique identifier
     * @return The agent ID
     */
    std::string getId() const;

    /**
     * @brief Perceive the environment and update internal state
     * @param observations The observations from the environment
     */
    virtual void perceive(const std::unordered_map<std::string, double>& observations) = 0;

    /**
     * @brief Select an action based on current internal state
     * @return The selected action
     */
    virtual std::string selectAction() = 0;

    /**
     * @brief Update the agent's internal state based on reward
     * @param reward The reward received from the environment
     */
    virtual void update(double reward) = 0;

protected:
    std::string id_;
    std::unordered_map<std::string, double> state_;
};

/**
 * @brief Reactive agent that maps perceptions directly to actions
 */
class ReactiveAgent : public Agent {
public:
    ReactiveAgent(const std::string& id);
    ~ReactiveAgent() override = default;

    void perceive(const std::unordered_map<std::string, double>& observations) override;
    std::string selectAction() override;
    void update(double reward) override;

    /**
     * @brief Add a rule to the agent's rule base
     * @param condition The condition function
     * @param action The action to take when condition is true
     */
    void addRule(std::function<bool(const std::unordered_map<std::string, double>&)> condition,
                 const std::string& action);

private:
    struct Rule {
        std::function<bool(const std::unordered_map<std::string, double>&)> condition;
        std::string action;
    };

    std::vector<Rule> rules_;
};

/**
 * @brief Deliberative agent that uses internal models for planning
 */
class DeliberativeAgent : public Agent {
public:
    DeliberativeAgent(const std::string& id);
    ~DeliberativeAgent() override = default;

    void perceive(const std::unordered_map<std::string, double>& observations) override;
    std::string selectAction() override;
    void update(double reward) override;

    /**
     * @brief Set the agent's goal
     * @param goal The goal state
     */
    void setGoal(const std::unordered_map<std::string, double>& goal);

    /**
     * @brief Add a model of action effects
     * @param action The action
     * @param effects The effects function
     */
    void addActionModel(const std::string& action,
                        std::function<std::unordered_map<std::string, double>(
                            const std::unordered_map<std::string, double>&)> effects);

private:
    std::unordered_map<std::string, double> goal_;
    std::unordered_map<std::string,
                       std::function<std::unordered_map<std::string, double>(
                           const std::unordered_map<std::string, double>&)>> actionModels_;
};

/**
 * @brief Environment interface for multi-agent systems
 */
class Environment {
public:
    Environment();
    virtual ~Environment() = default;

    /**
     * @brief Add an agent to the environment
     * @param agent The agent to add
     */
    void addAgent(std::shared_ptr<Agent> agent);

    /**
     * @brief Get observations for a specific agent
     * @param agentId The agent ID
     * @return The observations for the agent
     */
    virtual std::unordered_map<std::string, double> getObservations(const std::string& agentId) = 0;

    /**
     * @brief Process an action from an agent
     * @param agentId The agent ID
     * @param action The action to process
     * @return The reward for the action
     */
    virtual double processAction(const std::string& agentId, const std::string& action) = 0;

    /**
     * @brief Run a simulation step for all agents
     */
    void step();

    /**
     * @brief Check if the environment has reached a terminal state
     * @return True if terminal, false otherwise
     */
    virtual bool isTerminal() = 0;

protected:
    std::unordered_map<std::string, std::shared_ptr<Agent>> agents_;
};

/**
 * @brief Communication channel for agent message passing
 */
class CommunicationChannel {
public:
    CommunicationChannel();
    ~CommunicationChannel() = default;

    /**
     * @brief Send a message from one agent to another
     * @param senderId The sender agent ID
     * @param receiverId The receiver agent ID
     * @param message The message content
     */
    void sendMessage(const std::string& senderId, const std::string& receiverId,
                     const std::string& message);

    /**
     * @brief Get all messages for a specific agent
     * @param agentId The agent ID
     * @return The messages for the agent
     */
    std::vector<std::pair<std::string, std::string>> getMessages(const std::string& agentId);

    /**
     * @brief Clear all messages in the channel
     */
    void clearMessages();

private:
    struct Message {
        std::string senderId;
        std::string receiverId;
        std::string content;
    };

    std::vector<Message> messages_;
};

/**
 * @brief Coordination mechanism for multi-agent systems
 */
class CoordinationMechanism {
public:
    CoordinationMechanism();
    virtual ~CoordinationMechanism() = default;

    /**
     * @brief Register an agent with the coordination mechanism
     * @param agentId The agent ID
     */
    void registerAgent(const std::string& agentId);

    /**
     * @brief Coordinate actions among registered agents
     * @param agentActions Map of agent IDs to their proposed actions
     * @return Map of agent IDs to their coordinated actions
     */
    virtual std::unordered_map<std::string, std::string> coordinate(
        const std::unordered_map<std::string, std::string>& agentActions) = 0;

protected:
    std::vector<std::string> registeredAgents_;
};

/**
 * @brief Task allocation mechanism for multi-agent systems
 */
class TaskAllocation : public CoordinationMechanism {
public:
    TaskAllocation();
    ~TaskAllocation() override = default;

    /**
     * @brief Add a task to the system
     * @param taskId The task ID
     * @param requirements The task requirements
     */
    void addTask(const std::string& taskId,
                 const std::unordered_map<std::string, double>& requirements);

    /**
     * @brief Set agent capabilities
     * @param agentId The agent ID
     * @param capabilities The agent capabilities
     */
    void setAgentCapabilities(const std::string& agentId,
                              const std::unordered_map<std::string, double>& capabilities);

    std::unordered_map<std::string, std::string> coordinate(
        const std::unordered_map<std::string, std::string>& agentActions) override;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, double>> tasks_;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> agentCapabilities_;
};

/**
 * @brief Learning algorithm for multi-agent systems
 */
class MultiAgentLearning {
public:
    MultiAgentLearning();
    virtual ~MultiAgentLearning() = default;

    /**
     * @brief Initialize the learning algorithm
     * @param numAgents The number of agents
     * @param stateSize The size of the state space
     * @param actionSize The size of the action space
     */
    virtual void initialize(int numAgents, int stateSize, int actionSize) = 0;

    /**
     * @brief Update the learning algorithm with new experiences
     * @param agentId The agent ID
     * @param state The current state
     * @param action The action taken
     * @param reward The reward received
     * @param nextState The next state
     * @param done Whether the episode is done
     */
    virtual void update(int agentId, const std::vector<double>& state, int action, double reward,
                        const std::vector<double>& nextState, bool done) = 0;

    /**
     * @brief Get the best action for a given state
     * @param agentId The agent ID
     * @param state The current state
     * @return The best action
     */
    virtual int getBestAction(int agentId, const std::vector<double>& state) = 0;

    /**
     * @brief Get an exploratory action for a given state
     * @param agentId The agent ID
     * @param state The current state
     * @param epsilon The exploration rate
     * @return The selected action
     */
    virtual int getExploratoryAction(int agentId, const std::vector<double>& state,
                                     double epsilon) = 0;
};

/**
 * @brief Independent Q-learning for multi-agent systems
 */
class IndependentQLearning : public MultiAgentLearning {
public:
    IndependentQLearning(double learningRate = 0.1, double discountFactor = 0.99);
    ~IndependentQLearning() override = default;

    void initialize(int numAgents, int stateSize, int actionSize) override;
    void update(int agentId, const std::vector<double>& state, int action, double reward,
                const std::vector<double>& nextState, bool done) override;
    int getBestAction(int agentId, const std::vector<double>& state) override;
    int getExploratoryAction(int agentId, const std::vector<double>& state,
                             double epsilon) override;

private:
    double learningRate_;
    double discountFactor_;
    std::vector<std::unordered_map<std::string, std::vector<double>>> qTables_;

    std::string stateToString(const std::vector<double>& state);
};

} // namespace mas
} // namespace phynexus

#endif // PHYNEXUS_MAS_H
