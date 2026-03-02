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

// The module class needs to be registered with OMNeT++
Define_Module(Broker);



/*
Initialise the Broker by suscribing to the Sender signal
*/
void Broker::initialize()
{
    getSimulation()->getSystemModule()->subscribe("registerAgent", this); // used to register stepping agents
    getSimulation()->getSystemModule()->subscribe("unregisterAgent", this);// used to unregister stepping agents
   
    allAgentsDone = false;

    // Lifted from stepper class
    getSimulation()->getSystemModule()->subscribe("senderToStepper", this); //suscribe to the broker's signal
    getSimulation()->getSystemModule()->subscribe("modifyStepSize", this);// used to modify the size of the step used by Stepper

   // todo: unsubscirbe
}

/*
    Detect if we just started an agent's STEP event. 
    If so, request observations from that agent via the pullObservations signal.
*/
void Broker::handleMessage(cMessage *msg)
{
    std::string messageName(msg->getName());
    // Upper layer has requested a step to take place
    if (messageName.find("STEP-") != std::string::npos) { // TODO: Make sure this check passes. Might need to do the getName() first or something
        std::string id = messageName.substr(messageName.find("-")+1); // grab agent ID from string "STEP-{idNumber}"
        emit(pullObservations, id.c_str()); 
    } else {
        // This is likely an end-of-step event (EOS). Do nothing, the upper layer will handle it.
    }
}

/*
    Agent registration/unregistration, and step size modification
    Adds/removes agents from the list of active agents, and schedules STEP events as requested.
*/
void Broker::receiveSignal(cComponent *source, simsignal_t signalID, const char *value, cObject *obj){
    Enter_Method("schedule a step event(self message)"); //need this for direct messaging. Allows us to call scheduleMessage from the sender(ownership).
    const char *signalName = getSignalName(signalID);

    if (strcmp(signalName, "registerAgent") == 0) {
        // BROKER PORTION ---------------------------------------------------------------------
        EV_TRACE << "Registering new agent with Broker..." << std::endl;
        
        std::string id(value);

        EV_TRACE << "Agent ID: " << id << std::endl;

        // Creating detailsfor new agent
        BrokerDetails EOS_details;
        EOS_details.isReset = true;
        EOS_details.endOfStep = new cMessage((std::string("EOS-") + id).c_str()); // Name of the message should match EOS-<ID>
        EOS_details.rlId = id;

        //Inserting new agent details into map (but do not schedule it yet!)
        activeAgents.insert({id,EOS_details});

        // STEPPER PORTION --------------------------------------------------------------------
        EV_TRACE << "Registering new agent with Stepper..." << std::endl;

        cSimTime * stepSize = (cSimTime *) obj;

        // Creating detailsfor new agent
        StepDetails STEP_details;
        STEP_details.isReset = true;
        STEP_details.stepMsg = new cMessage((std::string("STEP-") + id).c_str()); // Name of the message should match STEP-<ID>
        STEP_details.stepSize = stepSize->simtime.dbl();
        STEP_details.rlId = id;

        delete stepSize;

        //Inserting new agent details into map
        activeAgentsStepper.insert({id,STEP_details});

        //Schedule first step
        scheduleAt(simTime() + STEP_details.stepSize, STEP_details.stepMsg);
        EV_TRACE << "Agent " << id << " will step in " <<  STEP_details.stepSize << " seconds at " << simTime() + STEP_details.stepSize << std::endl;
    } else if (strcmp(signalName, "unregisterAgent") == 0){
        //Get id
        std::string id(value);

        // BROKER PORTION -----------------------------------------------
        //Set done for agent to true and step right away
        activeAgents[id].done = true;
        
        if(activeAgents[id].endOfStep->isScheduled())
            cancelEvent(activeAgents[id].endOfStep);

        // Schedule an EOS message for the specific agent.
        scheduleAt(simTime(), activeAgents[id].endOfStep);

        //Check if all agents are done and store in variable for SimulationRunner
        allAgentsDone = areAllAgentsDone();

        // STEPPER PORTION ---------------------------------------------

        EV_TRACE << "Removing agent from Stepper map: " << id << std::endl;

        // Remove agent details from map after deleting the msg
        if (activeAgentsStepper[id].stepMsg->isScheduled()){
            cancelEvent(activeAgentsStepper[id].stepMsg);
            take(activeAgentsStepper[id].stepMsg);
        }
        delete activeAgentsStepper[id].stepMsg;

        activeAgentsStepper.erase(id);
    } else if (strcmp(signalName, "modifyStepSize") == 0) {
        // STEPPER: schedule a new step event
        //Get id
        std::string id(value);
        float stepSize = (float) ((cSimTime *) obj)->simtime.dbl();
        activeAgentsStepper[id].stepSize = stepSize;

        if(activeAgentsStepper[id].stepMsg->isScheduled()){
            cancelEvent(activeAgentsStepper[id].stepMsg);
            take(activeAgentsStepper[id].stepMsg);
            scheduleAt(simTime() + activeAgentsStepper[id].stepSize, activeAgentsStepper[id].stepMsg);
        }
    }


}


bool Broker::areAllAgentsDone(){
    bool allDone = true;
    for (auto& it: activeAgents) {
        if(!it.second.done)
            allDone = false;
     }
    return allDone;
}


/*

*/
void Broker::receiveSignal(cComponent *source, simsignal_t signalID, cObject *value, cObject *obj)
{
    Enter_Method("schedule a step event(self message)"); //need this for direct messaging. Allows us to call scheduleMessage from the sender(ownership).
    const char *signalName = getSignalName(signalID);
    // BROKER SECTION

    
    // Agent has sent an observation. Notify the broker.
    if (strcmp(signalName, "senderToStepper") == 0)
    {
        //Get id
        cString *c_id = (cString *) obj;
        std::string id = c_id->str; // segfault if signal did not include sender ID in the details.
        BrokerData *data = (BrokerData *)value;
        
        EV_TRACE << "Received signal senderToStepper from "<< id << std::endl;
        // automatically schedule another event (eventually remove this?)
        if (!data->getDone()){
            cancelEvent(activeAgentsStepper[id].stepMsg);
            take(activeAgentsStepper[id].stepMsg);
            scheduleAt(simTime() + activeAgentsStepper[id].stepSize, activeAgentsStepper[id].stepMsg);
        }
        //CHeck if ORCA's obs is vlid, otherwise skip this step
        if(data->isValid()){
            activeAgents[id].observation = data->getObs();

            if (!data->isReset()){
                activeAgents[id].reward = data->getReward();
                activeAgents[id].done = data->getDone();
            }
            bool test = activeAgents[id].endOfStep->isScheduled(); // segfault
            if(test) { 
                cancelEvent(activeAgents[id].endOfStep);
            }
            // Schedule an EOS message for the specific agent.
            scheduleAt(simTime(), activeAgents[id].endOfStep);


            EV_TRACE <<  simTime() <<" Scheduled end of step for " << id << "..." << std::endl;
            
        }
        //Clean up
        delete obj;
        delete value;
    } 
    else  
    {
        EV_TRACE << "Signal received by Broker not recognised" << std::endl;
    }
}


/*
    Called by the upper layer. 
    Forwards provided actions to every agent in the list (via the stepper, for now)
*/
void Broker::setActionAndMove(std::unordered_map<std::string, std::tuple<ActionType, bool>> &actionsAndMoves)
{
    BrokerData *data;
    cString *obj;
    // Build a BrokerData for each agent, containing instructions to reset, take an action, etc.
    for (auto& it: actionsAndMoves) {
        data = new BrokerData();
        if (std::get<1>(it.second)){
            data->setReset(std::get<1>(it.second));
        }
        else
        {
            data->setAction(std::get<0>(it.second));
            data->setReset(false);
        }
        // Forward instructions to the current agent
        obj = new cString(it.first);
        emit(actionResponse, data, obj); // Also pass the agent name (obj)
        delete data;
        delete obj;
    }
}



Broker::~Broker() {
    for (auto& it: activeAgentsStepper) {
        // Get agent id
        if (it.second.stepMsg->isScheduled()){
            cancelEvent(it.second.stepMsg);
            take(it.second.stepMsg);
        }
            delete it.second.stepMsg;
     }
}

void Broker::finish(){
     for (auto& it: activeAgents) {
        // Get agent id
        if (it.second.endOfStep->isScheduled()){
            cancelEvent(it.second.endOfStep);
        }
        delete it.second.endOfStep;
     }
}









// Getters (for the trainer)
// Observations
ObsType Broker::getObservation(std::string id){
    return activeAgents[id].observation;
}

std::unordered_map<std::string, ObsType> Broker::getObservations(){
    std::unordered_map<std::string, ObsType> observations;
    for (auto& it: activeAgents) {
        // Get agent id
        std::string id = it.first;
        auto observation = it.second.observation;
        observations.insert({id, observation});
    }
    return observations;
}

// Rewards
RewardType Broker::getReward(std::string id){
    return activeAgents[id].reward;
}

std::unordered_map<std::string, RewardType> Broker::getRewards(){
    std::unordered_map<std::string,RewardType> rewards;
    for (auto& it: activeAgents) {
        // Get agent id
        std::string id = it.first;
        auto reward = it.second.reward;
        rewards.insert({id, reward});
    }
    return rewards;
}

// Dones
bool Broker::getDone(std::string id)
{
    return activeAgents[id].done;
}

std::unordered_map<std::string, bool> Broker::getDones(){
    std::unordered_map<std::string, bool> dones;
    for (auto& it: activeAgents) {
        // Get agent id
        std::string id = it.first;
        auto done = it.second.done;
        dones.insert({id, done});
    }
    return dones;
}

bool Broker::getAllDone(){
    return allAgentsDone;
}
