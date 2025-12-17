#include "JamesCC.h"

namespace learning{
Register_Class(JamesCC); // Lets omnet see and use this class
JamesCC::JamesCC() :
        TcpNewReno(), RLInterface() {
}

JamesCC::~JamesCC() {
    getSimulation()->getSystemModule()->unsubscribe(stringId.c_str(), (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("actionResponse", (cListener*) this);
}

// RayNet: Called to initalize the agent
void JamesCC::initialize()
{   
    int _stateSize = this->conn->getTcpMain()->par("stateSize");
    int _maxObsCount = this->conn->getTcpMain()->par("maxObsCount");
    setOwner((cComponent*) conn->getTcpMain());
    RLInterface::initialize(_stateSize, _maxObsCount);
    TcpNewReno::initialize();
    // Assign this RayNet agent an ID (shuld this be done some time after initialization to ensure we are ready to step?)
    std::string s("JamesCC");
    this->setStringId(s);
    initMsg = new cMessage("JAMESCC-INIT"); 
    conn->scheduleAt(simTime() + 1, initMsg);
}

void JamesCC::handleMessage(cMessage *msg)
{
    if(msg->isSelfMessage()){
    conn->scheduleAt(simTime() + 1, initMsg);
    
    if(!isRegistered){
        isRegistered = true;
        cObject* simtime = new cSimTime(1);
        this->setOwner(conn);
        RLInterface::initialise();

        // Generate ID for this agent
        std::string s("JamesCC");
        this->setStringId(s);



        conn->emit(this->registerSig, stringId.c_str(), simtime);
    }
    }
    else{
        EV_DEBUG << "Stepper should only receive self messages!" << std::endl;
    }
}

void JamesCC::step(ActionType action)
{
}

void JamesCC::cleanup()
{
}

ObsType JamesCC::random()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.05, 0.05);
    ObsType values;
    for (int n = 0; n < 4; ++n)
        values[n] = dis(gen);
    return values;
}

void  JamesCC::decisionMade(ActionType action){
    if(this->isReset){
        //Reset state for next iterations
        steps_beyond_done = -10;
        state = random();
        std::cout << "Environment reset!" << std::endl;
    }else{
        step(action);
    }

} // defines what to do when decision is made
ObsType JamesCC::getRLState(){
    return state;
}

RewardType JamesCC::getReward(){
    RewardType reward;
    reward = 1.0;
    return reward;
}
bool JamesCC::getDone(){
    bool done = false;

    if (false) // some condition to check if the simulation is done
    {
        done = true;
    }

    return done;

}
void JamesCC::resetStepVariables()
{
}

ObsType JamesCC::computeObservation(){
    return getRLState();

}
RewardType JamesCC::computeReward(){
    return getReward();
}

void JamesCC::finish(){

}
}