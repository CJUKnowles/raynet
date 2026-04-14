/*
 * Broker.cpp
 *
 *  Created on: Sep 29, 2021
 *      Author: basil
 * 
 *  Stepper and Broker combined: Mar 02, 2026
 *      Author: James
 */

#include "Broker.h"
#include "typedefs.h"

Define_Module(Broker);

void Broker::initialize()
{
    std::string obsModeStr = par("obsCollectionMode").stringValue();
    if (obsModeStr == "IMMEDIATE") obsCollectionMode = IMMEDIATE;
    else if (obsModeStr == "GROUPED") obsCollectionMode = GROUPED;
    else if (obsModeStr == "INTERVALED") obsCollectionMode = INTERVALED;
    else throw cRuntimeError("Invalid mode: %s", obsModeStr.c_str());
    cout << "Broker ObsCollectionMode: " << obsModeStr << endl;
    
    getSimulation()->getSystemModule()->subscribe("registerAgent", this); // used to register stepping agents
    getSimulation()->getSystemModule()->subscribe("unregisterAgent", this);// used to unregister stepping agents
    getSimulation()->getSystemModule()->subscribe("obsResponse", this); //suscribe to the broker's signal
    getSimulation()->getSystemModule()->subscribe("modifyStepSize", this);// used to modify the size of the step used by Stepper
}

// Clean up this component. Called at the end of the simulation.
void Broker::finish(){
    // Cancel and delete any remaining STEP or EOS events
    for (auto& it: activeAgents) {
        if (it.second.STEPmsg->isScheduled()) {
            cancelEvent(it.second.STEPmsg);
            take(it.second.STEPmsg);
        }
    
        delete it.second.STEPmsg;
    }

    if (EOSmsg->isScheduled()) {
        cancelEvent(EOSmsg);
    }
    delete EOSmsg;

    getSimulation()->getSystemModule()->unsubscribe("registerAgent", this); // used to register stepping agents
    getSimulation()->getSystemModule()->unsubscribe("unregisterAgent", this);// used to unregister stepping agents
    getSimulation()->getSystemModule()->unsubscribe("obsResponse", this); //suscribe to the broker's signal
    getSimulation()->getSystemModule()->unsubscribe("modifyStepSize", this);// used to modify the size of the step used by Stepper
}

/*
    Detect if we just started an agent's STEP event. 
    If so, request observations from that agent via the pullObservations signal.
*/
void Broker::handleMessage(cMessage *msg)
{
    std::string messageName(msg->getName());
    // Upper layer has requested a step to take place
    if (messageName.find("STEP-") != std::string::npos) {
        std::string agentID = messageName.substr(messageName.find("-")+1);
        emit(this->obsRequestSig, agentID.c_str()); 
    } else if (messageName.find("EOS") != std::string::npos) {
        EV_TRACE << "EOS event detected, doing nothing. " << std::endl;
    }
}

/*
    Agent registration/unregistration, and step size modification
    Adds/removes agents from the list of active agents, and schedules STEP events as requested.
    TODO: Perform registration with a signal that does not include simtime? Currently unused.
*/
void Broker::receiveSignal(cComponent *source, simsignal_t signalID, const char *value, cObject *obj){
    Enter_Method("schedule a step event(self message)"); //need this for direct messaging. Allows us to call scheduleMessage from the sender(ownership).
    const char *signalName = getSignalName(signalID);

    if (strcmp(signalName, "registerAgent") == 0) {
        // BROKER PORTION ---------------------------------------------------------------------
        EV_TRACE << "Registering new agent with Broker..." << std::endl;
        
        std::string id(value);

        EV_TRACE << "Agent ID: " << id << std::endl;

        // Creating details for new agent
        BrokerDetails details;
        details.rlId = id;
        details.isReset = true;
        details.STEPmsg = new cMessage((std::string("STEP-") + id).c_str()); // Name of the message should match STEP-<ID>

        //Inserting new agent details into map (but do not schedule it yet!)
        activeAgents.insert({id,details});
    } else if (strcmp(signalName, "unregisterAgent") == 0){
        //Get id
        std::string id(value);

        // BROKER PORTION -----------------------------------------------
        //Set done for agent to true and step right away
        activeAgents[id].done = true;
        
        if (activeAgents[id].STEPmsg->isScheduled()){
            cancelEvent(activeAgents[id].STEPmsg);
            take(activeAgents[id].STEPmsg);
        }
        delete activeAgents[id].STEPmsg;
        // Schedule an EOS message for the specific agent.
        if (this->obsCollectionMode == IMMEDIATE) {
            scheduleAt(simTime(), this->EOSmsg);
        }
        activeAgents.erase(id);
        //Check if all agents are done and store in variable for SimulationRunner
        this->allAgentsDone = areAllAgentsDone();
    } else if (strcmp(signalName, "modifyStepSize") == 0) {
        // STEPPER: schedule a new step event
        //Get id
        std::string id(value);

        if(activeAgents[id].STEPmsg->isScheduled()){
            cancelEvent(activeAgents[id].STEPmsg);
            take(activeAgents[id].STEPmsg);
        }
        scheduleAt(simTime() + ((cSimTime*) obj)->simtime, activeAgents[id].STEPmsg);
    }
}

/*
    Receives observation signals from agents.
    Stores them for later processing, and ends the current step.
*/
void Broker::receiveSignal(cComponent *source, simsignal_t signalID, cObject *value, cObject *obj)
{
    Enter_Method("schedule a step event(self message)"); //need this for direct messaging. Allows us to call scheduleMessage from the sender(ownership).
    const char *signalName = getSignalName(signalID);

    // Agent has sent an observation
    if (strcmp(signalName, "obsResponse") == 0)
    {
        std::string id = ((cString*) obj)->str; // segfault if signal did not include sender ID in the details.
        EV_TRACE << "Received signal obsResponse from "<< id << std::endl;

        // End this step if the data is valid (?)
        BrokerData* agentData = (BrokerData*) value;
        if(agentData->isValid()) {
            // Collect and store the agent's data
            activeAgents[id].observation = agentData->getObs();
            activeAgents[id].uncollected = true;    // This obs is now new an so far uncollected by RayNet
            if (!agentData->isReset()) {
                activeAgents[id].reward = agentData->getReward();
                activeAgents[id].done = agentData->getDone();
                if (agentData->getDone()) {
                    this->allAgentsDone = areAllAgentsDone();
                }
            }

            // If obs collection mode is IMMEDIATE, then every STEP should immediately trigger observation collection via an EOS event
            if (this->obsCollectionMode == IMMEDIATE) {
                scheduleAt(simTime(), this->EOSmsg);
            } else if (this->obsCollectionMode == GROUPED) {
                if (this->areAllObsUncollected()) {
                    scheduleAt(simTime(), this->EOSmsg);
                }
            }
            EV_TRACE << simTime() <<" Scheduled end of step for " << id << "..." << std::endl;
        }
        
        // Cleanup
        delete obj;
        delete value;
        return;
    } 
    EV_TRACE << "Signal received by Broker not recognised" << std::endl;
}

/*
    Forwards actions to all relevant agents.
    Called by the upper layer upon receiving an observation (rllib->GymApi->cmdrlenv->this). 
*/
void Broker::setActionAndMove(std::unordered_map<std::string, std::tuple<ActionType, bool>> &actionsAndMoves)
{
    // Build a BrokerData containing instructions to reset, take an action, etc. for each agent
    BrokerData *data;
    for (auto& it: actionsAndMoves) {
        data = new BrokerData();
        if (std::get<1>(it.second)){
            data->setReset(std::get<1>(it.second));
        } else {
            data->setAction(std::get<0>(it.second));
            data->setReset(false);
        }
        // Forward instructions to the current agent
        emit(this->performActionSig, data, new cString(it.first)); // Also pass the agent name (obj)
        delete data;
    }
}


// MARK: Getters and state management
// --------------------------------------------------------------------------------------------------------------------------------

// Compiles a map of all uncollected observations and returns it
std::unordered_map<std::string, ObsType> Broker::getObservations(){
    std::unordered_map<std::string, ObsType> observations;
    for (auto& it: activeAgents) {
        std::string id = it.first;              // key
        BrokerDetails obsDetails = it.second;   // value
        if (obsDetails.uncollected) {
            observations.insert({id, obsDetails.observation});
        }
    }
    return observations;
}



// Compiles a map of all uncollected rewards and returns it
std::unordered_map<std::string, RewardType> Broker::getRewards(){
    std::unordered_map<std::string, RewardType> rewards;
    for (auto& it: activeAgents) {
        std::string id = it.first;              // key
        BrokerDetails obsDetails = it.second;   // value
        if (obsDetails.uncollected) {
            rewards.insert({id, obsDetails.reward});
        }
    }
    return rewards;
}


// Compiles a map of all uncollected dones and returns it
std::unordered_map<std::string, bool> Broker::getDones(){
    std::unordered_map<std::string, bool> dones;
    for (auto& it: activeAgents) {
        std::string id = it.first;              // key
        BrokerDetails obsDetails = it.second;   // value
        if (obsDetails.uncollected) {
            dones.insert({id, obsDetails.done});
        }
    }
    return dones;
}

// Sets every agent to uncollected=false. Should be used after gathering all relevant state and returning it to the trainer.
int Broker::invalidateOldStates() {
    int numStatesUpdated = 0;
    for (auto& it: activeAgents) {
        std::string id = it.first;              // key
        if (activeAgents[id].uncollected) {
            activeAgents[id].uncollected = false;
            numStatesUpdated++;
        }
    }
    return numStatesUpdated;
}

// Checks if ALL agents are done, regardless of how new/old their current state is
bool Broker::areAllAgentsDone(){
    bool allDone = true;
    for (auto& it: activeAgents) {
        if(!it.second.done)
            allDone = false;
     }
     cout << "Are all agents done: " << allDone << endl;
    return allDone;
}

// Checks if all ACTIVE (not done) agents have uncollected observations
bool Broker::areAllObsUncollected() {
    for (auto& it: activeAgents) {
        if(!it.second.done && !it.second.uncollected) {
            return false;
        }
    }
    return true;
}

// Simple getters ---

ObsType Broker::getObservation(std::string id){
    return activeAgents[id].observation;
}

RewardType Broker::getReward(std::string id){
    return activeAgents[id].reward;
}

bool Broker::getDone(std::string id)
{
    return activeAgents[id].done;
}

bool Broker::getAllDone(){
    return allAgentsDone;
}

