#include "omnetpp/ccomponent.h"
#include "omnetpp/cobject.h"
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
#include "training/Observer.h"

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
    // conn -> subscribe("globalStateResponse", (cListener*) this);
    getSimulation()->getSystemModule()->subscribe("globalStateResponse", this);

    // Initalize parent classes
    // RLInterface::initialize(_stateSize, _maxObsCount); // Deprecated initialization function. Delete this later.
    RLInterface::initialise();
    TcpNoCongestionControl::initialize();
    cout << "ID: " << std::to_string(owner->getId()) << endl;
    // Set the RL ID of this component (for use by the training script). Ensure this is unique for multi-agent environments (perhaps use the IP of the host?)
    std::string s("Astrea" + std::to_string(owner->getId()));
    //std::string s("Astrea12345");
    setStringId(s);
    
    // Register this agent with RayNet
    cObject* simtime = new cSimTime(this->conn->getTcpMain()->par("monitorIntervalDuration"));
    owner->emit(this->registerSig, stringId.c_str(), simtime);
    owner->emit(this->registerAstreaAgentSig, stringId.c_str(), simtime);
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
    // if (active) {
    //     std::string s("Astrea");
    //     setStringId(s);
    //     this->isActive = active;
    // }
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
    double delta_rexmit_count = state->rexmit_count - this->last_rexmit_count;
    this->astreaIntervalDuration = (simTime() - this->lastIntervalTime).dbl();

    if (delta_snd_una == 0) {
        // If no acks have been received, this obs is invalid. Schedule another step and skip the current one.
        scheduleNextStep(this->astreaIntervalDuration); // re-use the last interval duration. Prevents shrinking SRTT during congestion from oversaturing event queue with STEP events.
        //scheduleNextStep(state->srtt.dbl());
        return std::nullopt;
    }

    // Begin preparing a local state report to send to the Observer
    LocalState* localState = new LocalState();

    // Initialize empty obs to populate as we go
    double obs[7] = {0,0,0,0,0,0,};

    // Throughput: How many bytes were DELIVERED this interval (basically goodput?)
    this->astreaThroughput = delta_snd_una / this->astreaIntervalDuration;
    this->astreaMaxThroughput = std::max(this->astreaMaxThroughput, this->astreaThroughput);
    obs[0] = this->astreaThroughput / this->astreaMaxThroughput;
    obs[1] = this->astreaMaxThroughput;
    localState->throughput = this->astreaThroughput;
    localState->throughputRatio = obs[0];
    localState->maxThroughput = obs[1];

    // Latency: What is our RTT relative to the minimum observed
    this->astreaSRTT = state->srtt.dbl();
    this->astreaMinDelay = std::min(this->astreaMinDelay, this->astreaSRTT);
    obs[2] = state->srtt.dbl()/this->astreaMinDelay;
    obs[3] = this->astreaMinDelay;
    localState->latency = this->astreaSRTT;
    localState->minLatency = obs[3];

    // CWND: How does our current cwnd compare to the observed BDP
    obs[4] = state->snd_cwnd/(this->astreaMaxThroughput*this->astreaMinDelay);
    localState->cwnd = state->snd_cwnd;
    localState->cwndRatio = obs[4];

    // Lossrate: What percentage of bytes sent this interval were retransmissions
    this->astreaLossRate = delta_rexmit_count/this->astreaIntervalDuration;
    obs[5] = this->astreaLossRate/this->astreaMaxThroughput;
    localState->lossRate = this->astreaLossRate;
    localState->lossRateRatio = obs[5];

    // in-flight: How many bytes are currently sent but not ACKed
    double inflight = state->snd_max - state->snd_una;
    obs[6] = inflight/state->snd_cwnd;
    localState->inflight = inflight;
    localState->inflightRatio = obs[6];

    // Pacing rate: How quickly are we sending bytes to the TCP stack
    // obs[7] = state->paceRate / this->astreaMaxThroughput;


    // Delay Metric: The delay metric is treated as optimal if within the forgiveness window. Otherwise, have it slowly decrease as delay inflates.
    // this->astreaDelayMetric = 1.0; 
    // if (this->astreaSRTT > this->astreaMinDelay * this->rewardDelayForgiveness) {                                   
    //     this->astreaDelayMetric = this->astreaMinDelay * this->rewardDelayForgiveness / this->astreaSRTT;
    // }


    if(debug) {
        cout << "-" << endl;
        cout << "-" << endl;
        cout << "\t" << stringId << " State:" << endl;
            cout << "\t\tsnd_una: " << state->snd_una << endl;
            cout << "\t\tdelta_snd_una: " << delta_snd_una << endl;
            cout << "\t\tcwnd: " << state->snd_cwnd << endl;
            cout << "\t\tsnd_max: " << state->snd_max << endl;
            cout << "\t\tSRTT: " << state->srtt << endl;
        cout << "\tObservations:" << endl;
            cout << "\t\tThroughput ratio: \t" << obs[0] << endl;
            cout << "\t\tMax Throughput: \t" << obs[1] << endl;
            cout << "\t\tLatency Ratio: \t\t"  << obs[2] << endl;
            cout << "\t\tMin Latency: \t\t"    << obs[3] << endl;
            cout << "\t\tRelative cwnd: \t\t"  << obs[4]  << endl;
            cout << "\t\tLossrate: \t\t"       << obs[5]  << endl;
            cout << "\t\tInflight bytes: \t" << obs[6] << endl;
            // cout << "\t\tPacerate: \t"       << obs[7]  << endl;
        cout << "-" << endl;
        cout << "-" << endl;
    } 

    conn->emit(throughputSignal, this->astreaThroughput);
    conn->emit(srttSignal, state->srtt);
    conn->emit(cwndSignal, state->snd_cwnd);
    conn->emit(intervalDurationSignal, this->astreaIntervalDuration);

    // Send data to Observer before returning
    
    conn->emit(this->astreaStateReportSig, (cObject*) localState, new cString(stringId));

    scheduleNextStep(state->srtt.dbl());
    return ObsType{
            obs[0],     // Normalized throughput (thr/thr_max)
            obs[1],     // max throughput (raw)
            obs[2],     // latency ratio (lat/lat_min)
            obs[3],                                               // Min latency (raw)
            obs[4],   // relative cwnd (cwnd/(thr_max)*(lat_min))
            obs[5],                     // loss (lossrate/thr_max)
            obs[6],                                           // in-flight bytes (pkt_flight/cwnd)
            //obs[7],                          // pacing rate (prate/thr_max, will require tcpPaced and currently doesn't work)
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

    // Emit a signal requested global state/reward. LocalState here is a placeholder.
    LocalState* dummy = new LocalState();
    conn->emit(this->globalStateRequestSig, dummy, new cString(stringId));
    if(debug) cout << "\t\tReward: " << this->reward << endl;
    return this->reward; // Reward is automatically set upon receiving a globalStateResponse from the Observer
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void Astrea::decisionMade(ActionType action) {
    if (debug) cout << "\tAstrea: decisionMade()" << endl;

    // Calculate the newCwnd
    uint32_t newCwnd;
    if (action >= 0) {
        newCwnd = (double) state->snd_cwnd * (1.0+responsivenessCoefficient*action);
    } else {
        newCwnd = (double) state->snd_cwnd / (1.0-responsivenessCoefficient*action);
    }
    newCwnd = max(newCwnd, state->snd_mss);
    double multiplier = (double) newCwnd/state->snd_cwnd; // For debugging/plotting

    // Attempt to change cwnd and pacing rate
    if (this->takeActions && newCwnd <= 1000000) {
        if (debug) cout << "\t\tcwnd changed from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x)" << endl;
        state->snd_cwnd = newCwnd;
        // TODO: Pacing
        // this->astreaPaceRate = (double) state->snd_cwnd / state->srtt.dbl();  // Bytes/s
        // dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(1/orcaPaceRate); // 1/paceRate = intersendingtime
        owner->emit(actionSignal, multiplier); // Emit action for plotting
    } else {
        // Invalid. Skip this action entirely.
        if (debug) cout << "\t\t" << "NOT CHANGING cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x) NOT CHANGING !!!!!!" << endl;
    }

    if (debug) {
        // cout << "\t\t" << (this->takeActions) << endl;
        // // cout << "\t\t" << (this->first_slowstart_complete) << endl;
        // cout << "\t\t" << (this->rttReportCount > 0) << endl;
        // cout << "\t\t" << (newCwnd < 1000000) << endl;
        // cout << "-" << endl;
    }
}


void Astrea::resetStepVariables()
{
    if (debug) cout << "\tAstrea: resetStepVariables()" << endl;
    if (state->srtt.dbl() == 0) {
        return; // Skipping a step, dont reset step variables
    }
    this->last_snd_max = state->snd_max;
    this->last_snd_una = state->snd_una;
    this->last_rexmit_count = state->rexmit_count;
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