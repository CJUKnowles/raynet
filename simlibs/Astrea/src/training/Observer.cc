#include "Observer.h"
#include <algorithm>
#include <cmath>
#include <cobjects.h>

Define_Module(Observer);

// Called at start of simulation
void Observer::initialize()
{
    this->debug = par("printDebugMessages");
    this->LINK_DELAY = par("astreaBottleneckDelay");
    this->LINK_DELAY *= .001; // ms to s
    this->BUFFER_SIZE = par("astreaDataCapacity");
    this->BUFFER_SIZE /= 8.0; // bits to bytes
    this->BANDWIDTH = par("astreaBottleneckDatarate");
    this->BANDWIDTH *= 125000; // Megabits/s to bytes/s
    this->globalState = new GlobalState();

    getSimulation()->getSystemModule()->subscribe("registerAstreaAgent", this);     // used to register Astrea agents
    getSimulation()->getSystemModule()->subscribe("unregisterAstreaAgent", this);   // used to unregister Astrea agents
    getSimulation()->getSystemModule()->subscribe("astreaStateReport", this);       // used to report a given agent's current state
    getSimulation()->getSystemModule()->subscribe("globalStateRequest", this);      // used to request global state info (rewards) from the Observer
    
    if (debug) {
        cout << "\tObserver using following network parameters: " << endl;
        cout << "\t\t" << "BANDWIDTH: " << this->BANDWIDTH << endl;
        cout << "\t\t" << "LINK_DELAY: " << this->LINK_DELAY << endl;
        cout << "\t\t" << "BUFFER_SIZE: " << this->BUFFER_SIZE << endl;
    }
}

// Called at end of simulation
void Observer::finish(){
    // TODO: cleanup, delete messages and state entries
    getSimulation()->getSystemModule()->unsubscribe("registerAstreaAgent", this);     // used to register Astrea agents
    getSimulation()->getSystemModule()->unsubscribe("unregisterAstreaAgent", this);   // used to unregister Astrea agents
    getSimulation()->getSystemModule()->unsubscribe("astreaStateReport", this);       // used to report a given agent's current state
    getSimulation()->getSystemModule()->unsubscribe("globalStateRequest", this);      // used to request global state info (rewards) from the Observer
}

/*
    Astrea agent (de)registration. 
    Adds/removes a given agent from the list of agent observations used to compute global state.
    TODO: Used a better signal with less redudant arguments. You just need the agent ID and maybe a timestamp, nothing else.
*/


void Observer::receiveSignal(cComponent *source, simsignal_t signalID, const char *value, cObject *obj){
    Enter_Method("schedule a step event(self message)"); // TODO: Change this? 
    const char *signalName = getSignalName(signalID);

    if (strcmp(signalName, "registerAstreaAgent") == 0) {
        EV_TRACE << "Registering new agent with Observer..." << std::endl;
        std::string id(value);
        EV_TRACE << "Agent ID: " << id << std::endl;
        if (debug) cout << "OBSERVER: Registering " << id << endl;
        
        // Insert this agent into the Observer's agent list, with an empty history to populate later
        StateHistory newHistory;
        astreaAgents.insert({id, newHistory});

    } else if (strcmp(signalName, "unregisterAgent") == 0){
        // Remove the specified agent from the Observer's agent list.
        EV_TRACE << "Deregistering new agent with Observer..." << std::endl;
        std::string id(value);
        EV_TRACE << "Agent ID: " << id << std::endl;
        if (debug) cout << "OBSERVER: Deregistering " << id << endl;
        astreaAgents.erase(id);
    }
}

/*
    Receives a given agent's localState and adds it to their StateHistory.
    Note that the Observer now takes ownership of the LocalState. It is responsible for eventually freeing that memory.
*/
void Observer::receiveSignal(cComponent *source, simsignal_t signalID, cObject *value, cObject *obj)
{
    Enter_Method("schedule a step event(self message)"); // TODO: Change this?
    const char *signalName = getSignalName(signalID);

    // Agent has sent an observation
    if (strcmp(signalName, "astreaStateReport") == 0)
    {
        std::string id = ((cString*) obj)->str;
        LocalState* agentCurrentState = check_and_cast<LocalState*>(value);
        astreaAgents[id].addStateEntry(agentCurrentState);
        this->globalState->needsUpdating = true; // GlobalState is now out of date and needs updating
        return;
    // Agent has requested global state information (more specifically, a reward value)
    } else if (strcmp(signalName, "globalStateRequest") == 0) {
        std::string id = ((cString*) obj)->str;
        // cout << "OBSERVER: Received global state request from " << id << endl;
        if (globalState->needsUpdating) {
            computeGlobalState();
        }
        emit(this->globalStateResponseSig, globalState->reward, obj); // Placeholder, send computed state back to the requester
        return;
    }
    EV_TRACE << "Signal received by Observer not recognised" << std::endl;
}

// Loop through all agents' most recent state reports to update the global state
void Observer::computeGlobalState() {
    globalState->reset();

    double latencySum = 0;      // Total latency
    double cwndSum = 0;         // Total cwnd
    double lossSum = 0;         // Total number of lost bytes
    double lossRatioSum = 0;    // Sum of loss ratios (loss rate relative to current throughput, NOT max)
    double avgThroughputSum = 0; // Sum of all average throughputs (avg over each agent's entire state history)
    int numFlows = 0;

    for (auto& [agentId, stateHistory] : astreaAgents) {
        if (!stateHistory.history.empty()) {
            LocalState* localState = stateHistory.history.front(); // The most recent localState for the given agent
            
            double throughput = localState->throughput;
            globalState->ovrThroughput += throughput;
            globalState->minThroughput = std::min(globalState->minThroughput, throughput);
            globalState->maxThroughput = std::max(globalState->maxThroughput, throughput);
            avgThroughputSum += stateHistory.getAverageThroughput(); // Used for reward metrics

            double latency = localState->latency;
            latencySum += latency;

            double cwnd = localState->cwnd;
            cwndSum += cwnd;
            globalState->minCwnd = std::min(globalState->minCwnd, cwnd);
            globalState->maxCwnd = std::max(globalState->maxCwnd, cwnd);

            lossSum += localState->lossRate;
            if (localState->throughput > 0) {
                lossRatioSum += (localState->lossRate/localState->throughput);
            }

            numFlows++;
        }
    }

    if (numFlows < 1) {
        string error = "ERROR: cannot compute global state before any local states have been received! This shouldn't happen.";
        throw runtime_error(error);
    }

    // AVERAGE VALUES ======
    globalState->avgLatency = latencySum/numFlows;
    globalState->avgCwnd = cwndSum/numFlows;
    globalState->lossRatio = lossSum/numFlows;

    // REWARD METRICS ======
    globalState->reward = 0.0;

    // Throughput metric
    globalState->throughputMetric = globalState->ovrThroughput/this->BANDWIDTH;
    globalState->reward += this->throughputWeight * globalState->throughputMetric;

    // Latency Metric
    double latencyThreshold = (1.0 + delayCoeff)*this->LINK_DELAY;
    if(globalState->avgLatency > latencyThreshold) {
        globalState->latencyMetric = (globalState->avgLatency - latencyThreshold) * 1; // TODO: Multiply by the paceRate
    } else {
        // Treat latency as optimal if it falls below the threshold
        globalState->latencyMetric = 0;
    }
    globalState->reward -= this->latencyWeight * globalState->latencyMetric;

    // Loss metric
    globalState->lossMetric = lossRatioSum/numFlows;
    globalState->reward -= this->lossWeight * globalState->lossMetric;
    
    // Fairness Metric: 
    double globalAvgThroughput = avgThroughputSum/numFlows; // Average of all average throughputs (kill me)
    double fairnessNumerator = 0;
    for (auto& [agentId, stateHistory] : astreaAgents) {
        if (!stateHistory.history.empty()) {
            fairnessNumerator += std::pow(stateHistory.avgThroughput - globalAvgThroughput, 2.0);
        }
    }
    double fairnessDenominator = numFlows * std::pow(avgThroughputSum,2.0);
    if (fairnessDenominator == 0) {
        globalState->fairnessMetric = 0;
    } else {
        globalState->fairnessMetric = std::sqrt(fairnessNumerator/fairnessDenominator);
    }
    globalState->reward -= this->fairnessWeight * globalState->fairnessMetric;

    // Stability Metric: average stability of all active astrea flows
    double stabilitySum = 0;
    for (auto& [agentId, stateHistory] : astreaAgents) {
        if (!stateHistory.history.empty()) {
            stabilitySum += stateHistory.getStability();
        }
    }
    globalState->stabilityMetric = stabilitySum/numFlows;
    globalState->reward -= this->stabilityWeight * globalState->stabilityMetric;

    // Use the throughtputWeight to bound the reward value, as it defines the upper limit
    globalState->reward = max(-this->throughputWeight, min(this->throughputWeight, globalState->reward));
    globalState->minReward = std::min(globalState->minReward, globalState->reward);
    if (debug) globalState->printSummary();

    // Global state has been updated. Only re-compute if new observations arrive.
    globalState->needsUpdating = false;
}
