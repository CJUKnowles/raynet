#include "omnetpp/ccomponent.h"
#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/TcpPacedConnection.h"
#include "inet/transportlayer/tcp/flavours/TcpNoCongestionControl.h"
#include "transportlayer/tcp/flavours/TcpPacedFamily.h"
#include <numeric>
#include <optional>
#ifdef CLEANSLATE
#include "CleanSlate.h"
#include "typedefs.h"
#include <inet/common/INETDefs.h>

using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(CleanSlate); // Lets omnet see and use this class

CleanSlate::CleanSlate():
    TcpNoCongestionControl(), RLInterface() {
    if (debug) cout << "\tCleanSlate: Constructor called!";
}

CleanSlate::~CleanSlate() {
    if (debug) cout << "\tCleanSlate: Destructor method called. Goodbye.";
    getSimulation()->getSystemModule()->unsubscribe(stringId.c_str(), (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("performAction", (cListener*) this);
    
}


// // RayNet: Called to initalize the agent
void CleanSlate::initialize() {
    if (debug) cout << "\tCleanSlate initialize()" << endl;
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
    std::string s("CleanSlate");
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
void CleanSlate::established(bool active) {
    state->snd_cwnd = 6000;
    if (debug) cout << "\tCleanSlate: established()" << endl;
    TcpNoCongestionControl::established(active);
    //dynamic_cast<TcpPacedConnection*>(conn)->subscribe(dynamic_cast<TcpPacedConnection*>(conn)->retransmissionRateSignal, (cListener*) this);
    if (active) {
        std::string s("CleanSlate");
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
std::optional<ObsType> CleanSlate::computeObservation(){
    if (debug) cout << "\tCleanSlate: computeObservation()" << endl; 
    
    //dynamic_cast<TcpPacedConnection*>(conn)->computeRetransmissionRate(); // Updates this->retransmissionBytes via TcpPaced Connection
    double delta_snd_max = state->snd_max - this->last_snd_max;
    double delta_snd_una = state->snd_una - this->last_snd_una;
    this->cleanslateIntervalDuration = (simTime() - this->lastIntervalTime).dbl();

    if (delta_snd_una == 0) {
        // If no acks have been received, this obs is invalid. Schedule another step and skip the current one.
        scheduleNextStep(this->cleanslateIntervalDuration); // re-use the last interval duration. Prevents shrinking SRTT during congestion from oversaturing event queue with STEP events.
        //scheduleNextStep(state->srtt.dbl());
        return std::nullopt;
    }

    // Throughput: How many bytes were DELIVERED this interval (basically goodput?)
    this->cleanslateThroughput = delta_snd_una / this->cleanslateIntervalDuration;
    this->cleanslateMaxThroughput = std::max(this->cleanslateMaxThroughput, this->cleanslateThroughput);

    // Lossrate: What percentage of bytes sent this interval were retransmissions
    // this->cleanslateLossRate = 0.0;
    // if (this->retransmissionRate > 0.0) {  // Avoid division by 0
    //     double transmissionRate = delta_snd_max/this->cleanslateIntervalDuration; // How many non-retransmits occurred this interval
    //     this->cleanslateLossRate = this->retransmissionRate / (this->retransmissionRate + transmissionRate);
    // }

    // ACKed: How many bytes were ACKed this interval (basically raw goodput?)
    this->cleanslateACKTotal= delta_snd_una;
    this->maxACKTotal = std::max(this->maxACKTotal, this->cleanslateACKTotal);

    // SRTT: Smoothed round trip time. Already tracked by TCP.
    this->cleanslateSRTT = state->srtt.dbl();
    this->cleanslateMinDelay = std::min(this->cleanslateMinDelay, this->cleanslateSRTT);

    // CWND: Size of the congestion window. Already tracked by TCP.
    this->cleanslateCwnd = (double) state->snd_cwnd;
    this->maxCwnd = std::max(this->maxCwnd, this->cleanslateCwnd);

    // Delay Metric: The delay metric is treated as optimal if within the forgiveness window. Otherwise, have it slowly decrease as delay inflates.
    this->cleanslateDelayMetric = 1.0; 
    if (this->cleanslateSRTT > this->cleanslateMinDelay * this->rewardDelayForgiveness) {                                   
        this->cleanslateDelayMetric = this->cleanslateMinDelay * this->rewardDelayForgiveness / this->cleanslateSRTT;
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
            cout << "\t\tThroughput: " << this->cleanslateThroughput / this->cleanslateMaxThroughput << endl;
            cout << "\t\tACKs: " << this->cleanslateACKTotal /  state->snd_cwnd << endl;
            cout << "\t\tMTP Duration: " << this->cleanslateIntervalDuration << endl;
            cout << "\t\tSRTT: " << this->cleanslateMinDelay / this->cleanslateSRTT << endl;
            cout << "\t\tDelay Metric: " << this->cleanslateDelayMetric  << endl;
        cout << "-" << endl;
        cout << "-" << endl;
    } 

    conn->emit(throughputSignal, this->cleanslateThroughput);
    conn->emit(srttSignal, state->srtt);
    conn->emit(cwndSignal, state->snd_cwnd);
    conn->emit(intervalDurationSignal, this->cleanslateIntervalDuration);

    scheduleNextStep(state->srtt.dbl());
    return ObsType{this->cleanslateThroughput / this->cleanslateMaxThroughput,     // Normalized throughput
            this->cleanslateACKTotal /  state->snd_cwnd,              // Normalized ACKs count (maybe use tcp_cwnd? ask aiden)     
            this->cleanslateIntervalDuration,                         // Monitor interval duration
            this->cleanslateMinDelay / this->cleanslateSRTT,                // Normalized SRTT (delay)
            this->cleanslateDelayMetric                               // Normalized SRTT (possibly forgiven, if within the forgiveness window)
        };

    // return {delta_snd_una,                      // Throughput (number of bytes acked)
    //         this->cleanslateMaxThroughput,      // Max observed throughtput (number of bytes acked)   
    //         state->snd_cwnd,                    // Current cwnd
    //         this->maxCwnd,                      // Max observed cwnd
    //         state->srtt.dbl(),                  // current srtt
    //         this->cleanslateMinDelay,           // Min SRTT observed 
    //         this->cleanslateIntervalDuration,   // Monitor interval duration
    //     };
}

RewardType CleanSlate::computeReward(){
    if (debug) cout << "\tCleanSlate: computeReward()" << endl;
    double reward = this->cleanslateThroughput/this->cleanslateMaxThroughput*this->cleanslateDelayMetric;
    if (debug) cout << "\t" << reward << endl;
    return reward;
    //return(this->cleanslateThroughput/state->srtt);
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void CleanSlate::decisionMade(ActionType action) {
    if (debug) cout << "\tCleanSlate: decisionMade()" << endl;
    
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


void CleanSlate::resetStepVariables()
{
    if (debug) cout << "\tCleanSlate: resetStepVariables()" << endl;
    if (state->srtt.dbl() == 0) {
        return; // Skipping a step, dont reset step variables
    }
    this->last_snd_max = state->snd_max;
    this->last_snd_una = state->snd_una;
    this->lastIntervalTime = simTime();
}

// Returns true if the agent is reporting this episode as complete. (Pretty sure this is never called. Just set done to true directly during an RLStep.)
bool CleanSlate::getDone() {
    if (debug) cout << "CleanSlate getDone(): If you're seeing this, getDone() probably isn't deprecated.";
    bool done = RLStepsTaken > 1000;
    if (debug) cout << "\tCleanSlate: " << RLStepsTaken << " steps completed. Returning " << done << endl;
    return done;
}

// RayNet method: Called after simulation completion? Unsure how this differs from reset()
void CleanSlate::cleanup()
{
    if (debug) cout << "\tCleanSlate: cleanUp()" << endl;
}

ObsType CleanSlate::getRLState(){
    if (debug) cout << "\tCleanSlate: getRLState()" << endl;
    // Deprecated, remove this later
}

RewardType CleanSlate::getReward(){
    if (debug) cout << "\tCleanSlate: getReward()" << endl;
    // Deprecated, remove this later
}


#endif