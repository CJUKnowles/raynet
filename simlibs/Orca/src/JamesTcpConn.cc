#ifdef ORCA
#include "JamesTcpConn.h"
#include "inet/transportlayer/tcp/TcpConnection.h"
#include "typedefs.h"

Define_Module (JamesTcpConn);

JamesTcpConn::JamesTcpConn()
{
    cout << "\tJamesTcpConn created!" << endl;
    stepInterval = 10;
} 
//test 

JamesTcpConn::~JamesTcpConn()
{
}


// // RayNet: Called to initalize the agent
void JamesTcpConn::initialize() {
    cout << "\tJamesTcpConn initialize()" << endl;
    
    
    // Set initial state?
    state = ObsType({0.0, 0.0, 0.0, 0.0});
    isRegistered = false;

    monitorInterval = new cMessage("MONITORINTERVAL");
    scheduleAt(simTime() + stepInterval, monitorInterval);
    // Query qry(stringId, rlState);
    // this->emit(querySignal, &qry);
}

// Receive self messaged created via scheduleAfter() or scheduleAt(). Used to perpetually schedule future RL events.
void JamesTcpConn::handleMessage(cMessage *msg) {
    
    if(msg->isSelfMessage() && msg == monitorInterval) {
        cout << "\tJamesTcpConn: handleMessage() (new step from monitorInterval)" << endl;
        monitorInterval = new cMessage("MONITORINTERVAL");
        scheduleAt(simTime() + stepInterval, monitorInterval); // triggers unrecognized timer error. maybe needs to be caught by processTimer()?
        if (!isRegistered) {
            isRegistered = true;
            
            this->setOwner(this);
            
            int _stateSize = par("stateSize");;
            int _maxObsCount = par("maxObsCount");
            RLInterface::initialize(_stateSize, _maxObsCount);

            // Generate IJamesPlainCCD for this agent
            std::string s("JamesTcpConn"); // TODO: make unique for multiagent?
            this->setStringId(s);
            return;
        }
    }

    TcpConnection::handleMessage(msg);
}   
    

// Receive an action from the policy and perform it
void JamesTcpConn::decisionMade(ActionType action) {
    cout << "\tJamesTcpConn: decisionMade()" << endl;
    if(this->isReset){
        // Do some sort of reset/init? I assume this is handled in reset()
    }

    // Perform the action that was passed in
    // TODO: Set CWND or something similar

}



// Return a reward based on the current state (some function of throughput maybe?)
RewardType JamesTcpConn::computeReward(){
    cout << "\tJamesTcpConn: computeReward()" << endl;
    return getReward();
}

// Return a reward based on the current state (some function of throughput maybe?)
RewardType JamesTcpConn::getReward() {
    cout << "\tJamesTcpConn: getReward()" << endl;
    RewardType reward;
    reward = 1.0; // placeholder

    return reward;
}

// Return the current state
ObsType JamesTcpConn::getRLState() {
    cout << "\tJamesTcpConn: getRLState()" << endl;
    return state;
}

// Returns true if the current simulation is complete
bool JamesTcpConn::getDone() {
    cout << "\tJamesTcpConn: getDone()" << endl;
    bool done = false;
    
    return false;
}

// idk!
void JamesTcpConn::resetStepVariables(){
    cout << "\tJamesTcpConn: resetStepVariables()" << endl;

}

// Make an observation and update the state?
ObsType JamesTcpConn::computeObservation() {
    cout << "\tJamesTcpConn: computeObservation()" << endl;
    return state;
}

// Cleanup operations after sim complete?
void JamesTcpConn::cleanup(){

}

// Question for self: what is the difference between computeObservation() and getRLState()? They both return the state anyway...
#endif