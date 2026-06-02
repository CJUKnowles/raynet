#include "omnetpp/ccomponent.h"
#include "omnetpp/cobject.h"
#include "omnetpp/simtime.h"
#include "omnetpp/simtime_t.h"
#include <optional>
#include "Astraea.h"
#include "typedefs.h"
#include <inet/common/INETDefs.h>
#include "training/Observer.h"

using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(Astraea); // Lets omnet see and use this class

Astraea::Astraea():
    TcpPacedNoCC(), RLInterface() {
    if (debug) cout << "\tAstraea: Constructor called!";
    registerAstraeaAgentSig = owner->registerSignal("registerAstraeaAgent");
    AstraeaStateReportSig = owner->registerSignal("AstraeaStateReport");
    globalStateRequestSig = owner->registerSignal("globalStateRequest");
}

Astraea::~Astraea() {
    if (debug) cout << "\tAstraea: Destructor method called. Goodbye.";
    getSimulation()->getSystemModule()->unsubscribe(stringId.c_str(), (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("performAction", (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("globalStateResponse", this);
}


// // RayNet: Called to initalize the agent
void Astraea::initialize() {
    if (debug) cout << "\tAstraea initialize()" << endl;
    this->rewardDelayForgiveness = this->conn->getTcpMain()->par("rewardDelayForgiveness");
    this->rewardLossMultiplier = this->conn->getTcpMain()->par("rewardLossMultiplier");
    this->maxRLSteps = this->conn->getTcpMain()->par("maxRLSteps");
    debug = this->conn->getTcpMain()->par("printDebugMessages");
    takeActions = this->conn->getTcpMain()->par("takeActions");

    // provide the RLInterface with a cComponent API (to use signaling functionality)
    setOwner((cComponent*) conn->getTcpMain());
    
    // Initalize parent classes
    RLInterface::initialise();
    TcpPacedNoCC::initialize();

    // Signals
    throughputSignal = conn->registerSignal("throughput");
    srttSignal = conn->registerSignal("srtt");
    cwndSignal = conn->registerSignal("cwnd");
    intervalDurationSignal = conn->registerSignal("intervalDuration");
    actionSignal = conn->registerSignal("action");
    getSimulation()->getSystemModule()->subscribe("globalStateResponse", this);
}

// OMNet Method? Called after component initialization is complete?
void Astraea::established(bool active) {
    if (debug) cout << "\tAstraea: established()" << endl;
    TcpPacedNoCC::established(active);

    if (active) {
        this->isActive = active;
        cout << "ID: " << std::to_string(owner->getId()) << endl;
        // Set the RL ID of this component (for use by the training script). Ensure this is unique for multi-agent environments.
        std::string s("Astraea" + std::to_string(this->numAgents + 1));
        setStringId(s);
        
        // Register this agent with RayNet
        cObject* simtime = new cSimTime(this->conn->getTcpMain()->par("monitorIntervalDuration"));
        owner->emit(this->registerSig, stringId.c_str(), simtime);
        owner->emit(this->registerAstraeaAgentSig, stringId.c_str(), simtime);
        scheduleNextStep(this->fixedIntervalDuration);
    }
}







// Perform and observation and store the result into the provided vector (or append to it, if you're keeping history)
std::optional<ObsType> Astraea::computeObservation(){
    if (debug) cout << "\tAstraea: computeObservation()" << endl; 
    
    //dynamic_cast<TcpPacedConnection*>(conn)->computeRetransmissionRate(); // Updates this->retransmissionBytes via TcpPaced Connection
    double delta_snd_max = state->snd_max - this->last_snd_max;
    double delta_snd_una = state->snd_una - this->last_snd_una;
    double delta_rexmit_count = state->rexmit_count - this->last_rexmit_count;

    // Begin preparing a local state report to send to the Observer
    LocalState* localState = new LocalState();

    // Initialize empty obs to populate as we go
    double obs[8] = {0,0,0,0,0,0, 0};

    // Throughput: How many bytes were DELIVERED this interval (basically goodput?)
    this->AstraeaThroughput = delta_snd_una / this->fixedIntervalDuration;
    this->AstraeaMaxThroughput = std::max(this->AstraeaMaxThroughput, this->AstraeaThroughput);
    if (this->AstraeaMaxThroughput == 0) {
        obs[0] = 0;
    } else {
        obs[0] = this->AstraeaThroughput / this->AstraeaMaxThroughput;
    }
    obs[1] = this->AstraeaMaxThroughput;
    localState->throughput = this->AstraeaThroughput;
    localState->throughputRatio = obs[0];
    localState->maxThroughput = obs[1];

    // Latency: What is our RTT relative to the minimum observed
    this->AstraeaSRTT = state->srtt.dbl();
    if (this->AstraeaSRTT != 0) {
        this->AstraeaMinDelay = std::min(this->AstraeaMinDelay, this->AstraeaSRTT);
    }
    obs[2] = state->srtt.dbl()/this->AstraeaMinDelay;
    obs[3] = this->AstraeaMinDelay;
    localState->latency = this->AstraeaSRTT;
    localState->minLatency = obs[3];

    // CWND: How does our current cwnd compare to the observed BDP
    if (this->AstraeaMaxThroughput*this->AstraeaMinDelay == 0) {
        obs[4] = 0;
    } else {
        obs[4] = state->snd_cwnd/(this->AstraeaMaxThroughput*this->AstraeaMinDelay);
    }
    localState->cwnd = state->snd_cwnd;
    localState->cwndRatio = obs[4];

    // Lossrate: What percentage of bytes sent this interval were retransmissions
    this->AstraeaLossRate = delta_rexmit_count/this->fixedIntervalDuration;
    if (this->AstraeaMaxThroughput == 0) {
        obs[5] = 0;
    } else {
        obs[5] = this->AstraeaLossRate/this->AstraeaMaxThroughput;
    }
    localState->lossRate = this->AstraeaLossRate;
    localState->lossRateRatio = obs[5];

    // in-flight: How many bytes are currently sent but not ACKed
    double inflight = state->snd_max - state->snd_una;
    if (state->snd_cwnd == 0) {
        obs[6] = 0;
    } else {
        obs[6] = inflight/state->snd_cwnd;
    }
    localState->inflight = inflight;
    localState->inflightRatio = obs[6];

    // Pacing rate: How quickly are we sending bytes to the TCP stack
    double prate;
    if (dynamic_cast<TcpPacedConnection*>(conn)->intersendingTime.dbl() != 0) {
        prate = (1.0/dynamic_cast<TcpPacedConnection*>(conn)->intersendingTime.dbl()); // segments sent per sec
    } else {
        prate = 8192; // Arbitrarily large value. Pacerate is virtually infinite until it is set.
    }
    obs[7] = (prate * state->snd_mss) / this->AstraeaMaxThroughput; // convert prate to bytes/s, matching maxthruput's units
    localState->prate = prate; // Observer only uses prate as a multiplier for latency penalty. Use segments/s.

    if(debug) {
        cout << "-" << endl;
        cout << "-" << endl;
        cout << "\t" << stringId << " step #" << this->RLStepsTaken << ":" << endl;
            cout << "\t\tsimtime: " << simTime().dbl() << endl;
            cout << "\t\tsnd_una: " << state->snd_una << endl;
            cout << "\t\tdelta_snd_una: " << delta_snd_una << endl;
            cout << "\t\tcwnd: " << state->snd_cwnd << endl;
            cout << "\t\tsnd_max: " << state->snd_max << endl;
            cout << "\t\tSRTT: " << state->srtt << endl;
            cout << "\t\tIntersending Time (segments): " << dynamic_cast<TcpPacedConnection*>(conn)->intersendingTime.dbl();
            cout << "\t\tPacerate (segments): " << prate << endl;
            cout << "\t\tIntersending Time (bytes): " << dynamic_cast<TcpPacedConnection*>(conn)->intersendingTime.dbl() / state->snd_mss;
            cout << "\t\tPacerate (bytes): " << prate * state->snd_mss << endl;
        cout << "\tObservations:" << endl;
            cout << "\t\tThroughput ratio: \t" << obs[0] << endl;
            cout << "\t\tMax Throughput: \t" << obs[1] << endl;
            cout << "\t\tLatency Ratio: \t\t"  << obs[2] << endl;
            cout << "\t\tMin Latency: \t\t"    << obs[3] << endl;
            cout << "\t\tRelative cwnd: \t\t"  << obs[4]  << endl;
            cout << "\t\tLossrate: \t\t"       << obs[5]  << endl;
            cout << "\t\tInflight bytes: \t" << obs[6] << endl;
            cout << "\t\tPacerate: \t"       << obs[7]  << endl;
        cout << "-" << endl;
        cout << "-" << endl;
    } 

    conn->emit(throughputSignal, this->AstraeaThroughput);
    conn->emit(srttSignal, state->srtt);
    conn->emit(cwndSignal, state->snd_cwnd);

    // Send data to Observer before returning
    conn->emit(this->AstraeaStateReportSig, (cObject*) localState, new cString(stringId));

    // Update step count, and check if the step limit has been reached
    RLStepsTaken++;
    if (RLStepsTaken >= this->maxRLSteps) {
        done = true;
        cout << "\t" << stringId << " IS DONE AT STEP " << this->RLStepsTaken << endl;
    } else {
        // Finally, schedule the next step (will be automatically cancelled if done)
        scheduleNextStep(this->fixedIntervalDuration);
    }
    return ObsType({
            obs[0],     // Normalized throughput (thr/thr_max)
            obs[1],     // max throughput (raw)
            obs[2],     // latency ratio (lat/lat_min)
            obs[3],                                               // Min latency (raw)
            obs[4],   // relative cwnd (cwnd/(thr_max)*(lat_min))
            obs[5],                     // loss (lossrate/thr_max)
            obs[6],                                           // in-flight bytes (pkt_flight/cwnd)
            obs[7],                          // pacing rate (prate/thr_max, will require tcpPaced and currently doesn't work)
        });
}

RewardType Astraea::computeReward(){
    if (debug) cout << "\tAstraea: computeReward()" << endl;

    // Emit a signal requested global state/reward. LocalState here is a placeholder.
    LocalState* dummy = new LocalState();
    conn->emit(this->globalStateRequestSig, dummy, new cString(stringId));
    if(debug) cout << "\t\tReward: " << this->reward << endl;
    return this->reward; // Reward is automatically set upon receiving a globalStateResponse from the Observer
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void Astraea::decisionMade(ActionType action) {
    if (debug) cout << "\tAstraea: decisionMade()" << endl;

    // Calculate the newCwnd
    uint32_t newCwnd;
    if (action >= 0) {
        newCwnd = (double) state->snd_cwnd * (1.0+this->actionControlCoeff*action);
    } else {
        newCwnd = (double) state->snd_cwnd / (1.0-this->actionControlCoeff*action);
    }
    newCwnd = max(newCwnd, state->snd_mss);
    double multiplier = (double) newCwnd/state->snd_cwnd; // For debugging/plotting

    // Attempt to change cwnd and pacing rate (hard upper cwnd limit to prevent simulation slowdown during early training)
    if (this->takeActions && newCwnd <= 1000000) {
        // Action
        if (debug) cout << "\t\traw action: " << action << endl;
        if (debug) cout << "\t\tcwnd changed from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x)" << endl;
        state->snd_cwnd = newCwnd;
        owner->emit(actionSignal, multiplier); // Emit action for plotting

        // Pacing
        if (state->srtt.dbl() != 0) {
            this->AstraeaPaceRate = ((double) state->snd_cwnd / (double) state->snd_mss) / state->srtt.dbl();  // segments/s
            dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime((double) 1.0/AstraeaPaceRate); // seconds between each segment sent
            if (debug) cout << "\t\tprate set to " << this->AstraeaPaceRate << " (" << dynamic_cast<TcpPacedConnection*>(conn)->intersendingTime.dbl() << ")" << endl;
        } else {
            if (debug) cout << "\t\tno ACKS yet received, not setting pacerate" << endl;
        }
    } else {
        // Invalid. Skip this action entirely.
        if (debug) cout << "\t\t" << "NOT CHANGING cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x) NOT CHANGING !!!!!!" << endl;
    }
}


void Astraea::resetStepVariables()
{
    if (debug) cout << "\tAstraea: resetStepVariables()" << endl;
    this->last_snd_max = state->snd_max;
    this->last_snd_una = state->snd_una;
    this->last_rexmit_count = state->rexmit_count;
    this->lastIntervalTime = simTime();
}

// Returns true if the agent is reporting this episode as complete. (Pretty sure this is never called. Just set done to true directly during an RLStep.)
bool Astraea::getDone() {
    if (debug) cout << "Astraea getDone(): If you're seeing this, getDone() probably isn't deprecated.";
    bool done = RLStepsTaken > 1000;
    if (debug) cout << "\tAstraea: " << RLStepsTaken << " steps completed. Returning " << done << endl;
    return done;
}

// RayNet method: Called after simulation completion? Unsure how this differs from reset()
void Astraea::cleanup()
{
    if (debug) cout << "\tAstraea: cleanUp()" << endl;
}

ObsType Astraea::getRLState(){
    if (debug) cout << "\tAstraea: getRLState()" << endl;
    // Deprecated, remove this later
}

RewardType Astraea::getReward(){
    if (debug) cout << "\tAstraea: getReward()" << endl;
    // Deprecated, remove this later
}
