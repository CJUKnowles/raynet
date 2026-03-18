#include "omnetpp/ccomponent.h"
#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/TcpPacedConnection.h"
#include "inet/transportlayer/tcp/flavours/TcpNoCongestionControl.h"
#include "transportlayer/tcp/flavours/TcpPacedFamily.h"
#include <numeric>
#include <optional>
#ifdef ASTREA
#include "Astrea.h"
#include "typedefs.h"
#include <inet/common/INETDefs.h>

using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(Astrea); // Lets omnet see and use this class

Astrea::Astrea():
    TcpNoCongestionControl(), RLInterface() {
    if (debug) cout << "\tAstrea: Constructor called!";
}

Astrea::~Astrea() {
    if (debug) cout << "\tAstrea: Destructor method called. Goodbye.";
    getSimulation()->getSystemModule()->unsubscribe(stringId.c_str(), (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("performAction", (cListener*) this);
    
}


// // RayNet: Called to initalize the agent
void Astrea::initialize() {
    if (debug) cout << "\tAstrea initialize()" << endl;
    this->rewardDelayForgiveness = this->conn->getTcpMain()->par("rewardDelayForgiveness");
    this->rewardLossMultiplier = this->conn->getTcpMain()->par("rewardLossMultiplier");
    this->maxRLSteps = this->conn->getTcpMain()->par("maxRLSteps");
    debug = this->conn->getTcpMain()->par("printDebugMessages");
    takeActions = this->conn->getTcpMain()->par("takeActions");

    // provide the RLInterface with a cComponent API (to use signaling functionality)
    setOwner((cComponent*) conn->getTcpMain());
    
    // Initalize parent classes
    // RLInterface::initialize(_stateSize, _maxObsCount); // Deprecated initialization function. Delete this later.
    RLInterface::initialise();
    TcpNoCongestionControl::initialize();

    // Set the RL ID of this component (for use by the training script). Ensure this is unique for multi-agent environments (perhaps use the IP of the host?)
    std::string s("Astrea");
    setStringId(s);
    
    // Register this agent with RayNet
    cObject* simtime = new cSimTime(this->conn->getTcpMain()->par("monitorIntervalDuration"));
    owner->emit(this->registerSig, stringId.c_str(), simtime); 
    scheduleNextStep(this->initialStepLength);
    // Schedule the first RL step
    // RLStep = new cMessage("RLSTEP");
    // conn->scheduleAt(simTime() + RLStepInterval, RLStep);
}

// OMNet Method? Called after component initialization is complete?
void Astrea::established(bool active) {
    state->snd_cwnd = 6000;
    if (debug) cout << "\tAstrea: established()" << endl;
    TcpNoCongestionControl::established(active);
    //dynamic_cast<TcpPacedConnection*>(conn)->subscribe(dynamic_cast<TcpPacedConnection*>(conn)->retransmissionRateSignal, (cListener*) this);
    if (active) {
        std::string s("Astrea");
        setStringId(s);
        this->isActive = active;
    }
    throughputSignal = conn->registerSignal("throughput");
    srttSignal = conn->registerSignal("srtt");
    cwndSignal = conn->registerSignal("cwnd");
    intervalDurationSignal = conn->registerSignal("intervalDuration");
    actionSignal = conn->registerSignal("action");
}







// Perform and observation and store the result into the provided vector (or append to it, if you're keeping history)
std::optional<ObsType> Astrea::computeObservation(){
    if (debug) cout << "\tAstrea: computeObservation()" << endl; 
    
    //dynamic_cast<TcpPacedConnection*>(conn)->computeRetransmissionRate(); // Updates this->retransmissionBytes via TcpPaced Connection
    double delta_snd_max = state->snd_max - this->last_snd_max;
    double delta_snd_una = state->snd_una - this->last_snd_una;
    this->astreaIntervalDuration = (simTime() - this->lastIntervalTime).dbl();

    if (delta_snd_una == 0) {
        // If no acks have been received, this obs is invalid. Schedule another step and skip the current one.
        scheduleNextStep(this->astreaIntervalDuration); // re-use the last interval duration. Prevents shrinking SRTT during congestion from oversaturing event queue with STEP events.
        //scheduleNextStep(state->srtt.dbl());
        return std::nullopt;
    }

    // Throughput: How many bytes were DELIVERED this interval (basically goodput?)
    this->astreaThroughput = delta_snd_una / this->astreaIntervalDuration;
    this->astreaMaxThroughput = std::max(this->astreaMaxThroughput, this->astreaThroughput);

    // Lossrate: What percentage of bytes sent this interval were retransmissions
    // this->astreaLossRate = 0.0;
    // if (this->retransmissionRate > 0.0) {  // Avoid division by 0
    //     double transmissionRate = delta_snd_max/this->astreaIntervalDuration; // How many non-retransmits occurred this interval
    //     this->astreaLossRate = this->retransmissionRate / (this->retransmissionRate + transmissionRate);
    // }

    // ACKed: How many bytes were ACKed this interval (basically raw goodput?)
    this->astreaACKTotal= delta_snd_una;
    this->maxACKTotal = std::max(this->maxACKTotal, this->astreaACKTotal);

    // SRTT: Smoothed round trip time. Already tracked by TCP.
    this->astreaSRTT = state->srtt.dbl();
    this->astreaMinDelay = std::min(this->astreaMinDelay, this->astreaSRTT);

    // CWND: Size of the congestion window. Already tracked by TCP.
    this->astreaCwnd = (double) state->snd_cwnd;
    this->maxCwnd = std::max(this->maxCwnd, this->astreaCwnd);

    // Delay Metric: The delay metric is treated as optimal if within the forgiveness window. Otherwise, have it slowly decrease as delay inflates.
    this->astreaDelayMetric = 1.0; 
    if (this->astreaSRTT > this->astreaMinDelay * this->rewardDelayForgiveness) {                                   
        this->astreaDelayMetric = this->astreaMinDelay * this->rewardDelayForgiveness / this->astreaSRTT;
    }

    

    if(debug) {
        cout << "-" << endl;
        cout << "-" << endl;
        cout << "\tState:" << endl;
            cout << "\t\tsnd_una: " << state->snd_una << endl;
            cout << "\t\tdelta_snd_una: " << delta_snd_una << endl;
            cout << "\t\tcwnd: " << state->snd_cwnd << endl;
            cout << "\t\tsnd_max: " << state->snd_max << endl;
            cout << "\t\tSRTT: " << state->srtt << endl;
        cout << "\tObservations:" << endl;
            cout << "\t\tThroughput: " << this->astreaThroughput / this->astreaMaxThroughput << endl;
            cout << "\t\tACKs: " << this->astreaACKTotal /  state->snd_cwnd << endl;
            cout << "\t\tMTP Duration: " << this->astreaIntervalDuration << endl;
            cout << "\t\tSRTT: " << this->astreaMinDelay / this->astreaSRTT << endl;
            cout << "\t\tDelay Metric: " << this->astreaDelayMetric  << endl;
        cout << "-" << endl;
        cout << "-" << endl;
    } 

    conn->emit(throughputSignal, this->astreaThroughput);
    conn->emit(srttSignal, state->srtt);
    conn->emit(cwndSignal, state->snd_cwnd);
    conn->emit(intervalDurationSignal, this->astreaIntervalDuration);

    scheduleNextStep(state->srtt.dbl());
    return ObsType{this->astreaThroughput / this->astreaMaxThroughput,     // Normalized throughput
            this->astreaACKTotal /  state->snd_cwnd,              // Normalized ACKs count (maybe use tcp_cwnd? ask aiden)     
            this->astreaIntervalDuration,                         // Monitor interval duration
            this->astreaMinDelay / this->astreaSRTT,                // Normalized SRTT (delay)
            this->astreaDelayMetric                               // Normalized SRTT (possibly forgiven, if within the forgiveness window)
        };

    // return {delta_snd_una,                      // Throughput (number of bytes acked)
    //         this->astreaMaxThroughput,      // Max observed throughtput (number of bytes acked)   
    //         state->snd_cwnd,                    // Current cwnd
    //         this->maxCwnd,                      // Max observed cwnd
    //         state->srtt.dbl(),                  // current srtt
    //         this->astreaMinDelay,           // Min SRTT observed 
    //         this->astreaIntervalDuration,   // Monitor interval duration
    //     };
}

RewardType Astrea::computeReward(){
    if (debug) cout << "\tAstrea: computeReward()" << endl;
    double reward = this->astreaThroughput/this->astreaMaxThroughput*this->astreaDelayMetric;
    if (debug) cout << "\t" << reward << endl;
    return reward;
    //return(this->astreaThroughput/state->srtt);
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void Astrea::decisionMade(ActionType action) {
    if (debug) cout << "\tAstrea: decisionMade()" << endl;
    
    RLStepsTaken++;
    if (debug) cout << "\t\tRLSteps taken: " << RLStepsTaken << endl;
    if (RLStepsTaken >= this->maxRLSteps) {
            if (debug) cout << "\t\tWE ARE DONE! " << RLStepsTaken << " STEPS TAKEN!" << endl;
            done = true; // Don't set done yourself. Unsure of the correct way to handle this, but this isn't it.
    }

    double fakeAction = action;
    uint32_t newCwnd = ceil(std::pow(2.0, fakeAction) * (double) state->snd_cwnd);
    newCwnd =  max(state->snd_mss, newCwnd); // cwnd should not deflate below 1mss
    // dont let cwnd inflate to ridiculous values. Learning will take care of this eventually, but large values eventually kill simulations.
    if (newCwnd < 1000000) {
        conn->emit(actionSignal, std::pow(2.0, fakeAction));
        if (debug) cout << "\t\tChanging cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << (double)newCwnd/(double)state->snd_cwnd << "x)" << endl;
        if (takeActions) state->snd_cwnd = newCwnd;
    }
    

    double newIntersendingTime = state->srtt.dbl() / (double) state->snd_cwnd;  // Pace rate expressed as seconds between packets (cwnd/srtt per second)

}


void Astrea::resetStepVariables()
{
    if (debug) cout << "\tAstrea: resetStepVariables()" << endl;
    if (state->srtt.dbl() == 0) {
        return; // Skipping a step, dont reset step variables
    }
    this->last_snd_max = state->snd_max;
    this->last_snd_una = state->snd_una;
    this->lastIntervalTime = simTime();
}

// Returns true if the agent is reporting this episode as complete. (Pretty sure this is never called. Just set done to true directly during an RLStep.)
bool Astrea::getDone() {
    if (debug) cout << "Astrea getDone(): If you're seeing this, getDone() probably isn't deprecated.";
    bool done = RLStepsTaken > 1000;
    if (debug) cout << "\tAstrea: " << RLStepsTaken << " steps completed. Returning " << done << endl;
    return done;
}

// RayNet method: Called after simulation completion? Unsure how this differs from reset()
void Astrea::cleanup()
{
    if (debug) cout << "\tAstrea: cleanUp()" << endl;
}

ObsType Astrea::getRLState(){
    if (debug) cout << "\tAstrea: getRLState()" << endl;
    // Deprecated, remove this later
}

RewardType Astrea::getReward(){
    if (debug) cout << "\tAstrea: getReward()" << endl;
    // Deprecated, remove this later
}


#endif