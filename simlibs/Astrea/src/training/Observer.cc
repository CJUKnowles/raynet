#include "Observer.h"
#include <cobjects.h>

Define_Module(Observer);

// Called at start of simulation
void Observer::initialize()
{
    this->globalState = new GlobalState();
    getSimulation()->getSystemModule()->subscribe("registerAstreaAgent", this);     // used to register Astrea agents
    getSimulation()->getSystemModule()->subscribe("unregisterAstreaAgent", this);   // used to unregister Astrea agents
    getSimulation()->getSystemModule()->subscribe("astreaStateReport", this);       // used to report a given agent's current state
    getSimulation()->getSystemModule()->subscribe("globalStateRequest", this);      // used to request global state info (rewards) from the Observer
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
        cout << "OBSERVER: Registering " << id << endl;
        
        // Insert this agent into the Observer's agent list, with an empty history to populate later
        StateHistory newHistory;
        astreaAgents.insert({id, newHistory});

    } else if (strcmp(signalName, "unregisterAgent") == 0){
        // Remove the specified agent from the Observer's agent list.
        EV_TRACE << "Deregistering new agent with Observer..." << std::endl;
        std::string id(value);
        EV_TRACE << "Agent ID: " << id << std::endl;
        cout << "OBSERVER: Deregistering " << id << endl;
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
        this->globalState->needsUpdating = true; // Global state is no longer up-to-date. GlobalStateRequests will trigger it to update.
        
        // cout << "OBSERVER: Received state report from " << id << endl;
        return;
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

// // Returns the average throughput, sampled from the most recent throughput report of each active agent.
// // TODO: Use some sort of weighing of the previous X entries, like the paper
// double Observer::computeAverageThroughput() {
//     double sum = 0.0;
//     int count = 0;

//     for (auto& [agentId, stateHistory] : astreaAgents) {
//         if (!stateHistory.history.empty()) {
//             LocalState* latestState = stateHistory.history.front();
//             sum += latestState->throughput;
//             count++;
//         }
//     }

//     if (count == 0) {
//         return 0;
//     }

//     return sum/count;
// }

// Loop through all agents' most recent state reports to update the global state
void Observer::computeGlobalState() {
    globalState->reset();

    double latencySum = 0;
    double cwndSum = 0;
    double lossSum = 0;

    int numStates = 0;

    for (auto& [agentId, stateHistory] : astreaAgents) {
        if (!stateHistory.history.empty()) {
            LocalState* localState = stateHistory.history.front(); // The most recent localState for the given agent
            
            double throughput = localState->throughput;
            globalState->ovrThroughput += throughput;
            globalState->minThroughput = std::min(globalState->minThroughput, throughput);
            globalState->maxThroughput = std::max(globalState->maxThroughput, throughput);

            double latency = localState->delay;
            latencySum += latency;

            double cwnd = localState->cwnd;
            cwndSum += cwnd;
            globalState->minCwnd = std::min(globalState->minCwnd, cwnd);
            globalState->maxCwnd = std::max(globalState->maxCwnd, cwnd);

            double loss = localState->lossRate;
            lossSum += loss;

            numStates++;
        }
    }

    if (numStates > 0) {
        globalState->avgLatency = latencySum/numStates;
        globalState->avgCwnd = cwndSum/numStates;
        globalState->lossRatio = lossSum/numStates;
    }

    // TODO: Compute reward values like fairness
    globalState->reward = globalState->ovrThroughput; // PLACEHOLDER!!!

}