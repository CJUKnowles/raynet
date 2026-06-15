#include "RLInterface.h"
#include "omnetpp/ccomponent.h"
#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/TcpPacedConnection.h"
#include "transportlayer/tcp/TcpPaced.h"
#include "transportlayer/tcp/flavours/TcpCubic.h"
#include <algorithm>
#include <numeric>
#include <ostream>
#include "Orca.h"
#include "typedefs.h"
#include <inet/common/INETDefs.h>

using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(Orca); // Lets omnet see and use this class

namespace {
    std::string sanitizeAgentId(std::string id)
    {
        for (char& c : id) {
            if (!std::isalnum(static_cast<unsigned char>(c))) {
                c = '_';
            }
        }
        return id;
    }

    std::string makeOrcaAgentId(cComponent *owner)
    {
        cModule *tcpModule = dynamic_cast<cModule *>(owner);
        cModule *hostModule = tcpModule ? tcpModule->getParentModule() : nullptr;
        int hostIndex = hostModule ? hostModule->getIndex() : -1;

        if (hostIndex == 0) {
            return "Orca";
        }
        if (hostIndex > 0) {
            return "Orca" + std::to_string(hostIndex);
        }
        return "Orca_" + sanitizeAgentId(owner ? owner->getFullPath() : "unknown");
    }
}
Orca::Orca():
    TcpCubic(), RLInterface() {
    if (debug) cout << "\tOrca: Constructor called!";
}

Orca::~Orca() {
    if (debug) cout << "\tOrca: Destructor method called. Goodbye.";
    getSimulation()->getSystemModule()->unsubscribe(stringId.c_str(), (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("performAction", (cListener*) this);
}

// Called during sim initialization
void Orca::initialize() {
    if (debug) cout << "\tOrca initialize()" << endl;
    this->delayCoefficient = this->conn->getTcpMain()->par("delayCoefficient");
    this->lossCoefficient = this->conn->getTcpMain()->par("lossCoefficient");
    this->fixedIntervalDuration = this->conn->getTcpMain()->par("fixedIntervalDuration");
    this->takeActions = this->conn->getTcpMain()->par("takeActions");
    this->debug = this->conn->getTcpMain()->par("printDebugMessages");

    // provide the RLInterface with a cComponent API (for signals)
    RLInterface::setOwner((cComponent*) conn->getTcpMain());
    
    // Initalize parent classes
    RLInterface::initialise();
    TcpCubic::initialize();

    // Signals
    throughputSignal = owner->registerSignal("throughput");
    actionSignal = owner->registerSignal("action");
    TcpPacedConnection* pacedConn = dynamic_cast<TcpPacedConnection *>(conn);
    pacedConn->subscribe(pacedConn->retransmissionRateSignal, (cListener*) this);
}

// INET method, called after connection is established.
void Orca::established(bool active) {
    if (debug) cout << "\tOrca: established()" << endl;
    TcpCubic::established(active);
    
    // Only register this as an RL agent if this is a client
    if (active) {
        this->isActive = active;

        // Set the RayNet ID of this agent and register with the Broker
        RLInterface::setStringId(makeOrcaAgentId(owner));
        cObject* simtime = new cSimTime(0); // Used to contain initial step length, now deprecated.
        owner->emit(this->registerSig, stringId.c_str(), simtime); 

        // Finally, schedule this agent's first fixed-length RL step.
        RLInterface::scheduleNextStep(this->fixedIntervalDuration);

        // this->takeActions = false;
        this->debug = true;
    }
}

// Return the raw transport metrics used by the original Orca environment wrapper.
std::optional<ObsType> Orca::computeObservation(){
    if (debug) cout << "\t" << stringId << " computeObservation()" << endl; 

    // Collect interval-level delivery and loss metrics.
    double lastIntervalDuration = (simTime() - this->lastIntervalTime).dbl();
    TcpPacedConnection* pacedConnection = dynamic_cast<TcpPacedConnection*>(conn);
    this->bytesDelivered = pacedConnection->getDelivered();
    this->bytesLost = pacedConnection->getTotalDetectedLostBytes();
    this->delta_bytes_delivered = this->bytesDelivered - this->last_bytes_delivered;
    this->delta_bytes_lost = this->bytesLost - this->last_bytes_lost;
    this->orcaThroughput = lastIntervalDuration > 0.0 ? this->delta_bytes_delivered / lastIntervalDuration : 0.0;
    this->lossRate = lastIntervalDuration > 0.0 ? this->delta_bytes_lost / lastIntervalDuration : 0.0;
    this->delta_snd_max = state->snd_max - this->last_snd_max;
    this->delta_snd_una = state->snd_una - this->last_snd_una;
    this->delta_ack_cnt = state->ack_cnt - this->last_ack_cnt;

    // Convert current TCP values to the units used by the original Orca server.
    double averageDelayMs = this->rttReportCount > 0 ? (this->orcaDelaySum / this->rttReportCount) * 1000.0 : 0.0;
    double pacingRate = pacedConnection->intersendingTime.dbl() > 0.0 ? state->snd_mss / pacedConnection->intersendingTime.dbl() : 0.0;
    double cwndPackets = state->snd_mss > 0 ? state->snd_cwnd / (double)state->snd_mss : 0.0;
    double ssthreshPackets = state->snd_mss > 0 ? state->ssthresh / (double)state->snd_mss : 0.0;
    double packetsOut = state->snd_mss > 0 ? pacedConnection->getBytesInFlight() / (double)state->snd_mss : 0.0;
    double retransOut = state->snd_mss > 0 ? this->retransmissionRate * lastIntervalDuration / state->snd_mss : 0.0;
    double srttMs = state->srtt.dbl() * 1000.0;
    double minRttMs = this->rttReportCount > 0 ? this->orcaMinDelay * 1000.0 : 0.0;

    // Permanently hand congestion-window control to Orca after Cubic exits initial slow start.
    this->slowStartPassed = this->slowStartPassed || state->snd_cwnd > state->ssthresh;

    // Schedule the next raw metric collection interval.
    scheduleNextStep(this->fixedIntervalDuration);

    // Print the collected raw metrics when debug output is enabled.
    if(debug) {
        cout << "-" << endl;
        cout << "\taverageDelayMs: " << averageDelayMs << endl;
        cout << "\tthroughput: " << this->orcaThroughput << endl;
        cout << "\trttSamples: " << this->rttReportCount << endl;
        cout << "\tintervalDuration: " << lastIntervalDuration << endl;
        cout << "\tcwndPackets: " << cwndPackets << endl;
        cout << "\tpacingRate: " << pacingRate << endl;
        cout << "\tlossRate: " << this->lossRate << endl;
        cout << "\tsrttMs: " << srttMs << endl;
        cout << "\tminRttMs: " << minRttMs << endl;
    } 

    // Emit throughput for OMNeT++ result recording.
    owner->emit(throughputSignal, this->orcaThroughput);

    return ObsType({
        averageDelayMs,
        this->orcaThroughput,
        this->rttReportCount,
        lastIntervalDuration,
        50.0,
        cwndPackets,
        pacingRate,
        this->lossRate,
        srttMs,
        ssthreshPackets,
        packetsOut,
        retransOut,
        packetsOut,
        state->snd_mss,
        minRttMs
    });
}

RewardType Orca::computeReward(){
    return -1.0;
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void Orca::decisionMade(ActionType action) {
    if (debug) cout << "\t" << stringId << " decisionMade()" << endl;

    // Compute new cwnd from the given action (cwnd *= 2^action)
    double multiplier = std::pow(4.0, (double) action);
    uint32_t newCwnd = ceil(((double) state->snd_cwnd) * multiplier);
    newCwnd = max(newCwnd, state->snd_mss);
    
    // Apply actions only after Cubic exits slow start and the interval contains valid RTT data.
    if (this->takeActions && this->slowStartPassed && this->rttReportCount > 0) {
        if (debug) cout << "\t\tChanging cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x)" << endl;
        state->snd_cwnd = newCwnd;

        // Update pacing rate
        if(state->snd_cwnd > 0) {
            double paceFactor;
            if (state->snd_cwnd < state->ssthresh/2) {
                paceFactor = 2;
            }
            else{
                paceFactor = 1.2;
            }
        uint32_t maxWindow = std::max(state->snd_cwnd, dynamic_cast<TcpPacedConnection*>(conn)->getBytesInFlight());
        double pace = state->srtt.dbl()/(((double) (maxWindow) / (double)state->snd_mss) * paceFactor);
        dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(pace);
        }
        owner->emit(actionSignal, multiplier); // Emit action for plotting
    } else {
        // Invalid. Skip this action entirely.
        if (debug) {
            cout << "\t\t" << "NOT CHANGING cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x) NOT CHANGING !!!!!!" << endl;
        } 
    }
}


void Orca::resetStepVariables()
{
    if (debug) cout << "\t\t" << stringId << " resetStepVariables()" << endl;

    // Preserve unfinished interval metrics until a valid RTT-bearing observation is acknowledged.
    if (this->rttReportCount == 0) {
        return;
    }

    // Commit the valid interval boundary and begin collecting the next interval.
    this->orcaThroughput=0.0;    // The average delivery rate (throughput) over the last interval
    this->orcaDelaySum=0.0;      // Sum of all RTT reports received over an interval 
    this->orcaACKTotal=0.0;      // The number of valid acknowledgements over the last interval
    this->rttReportCount=0; // The number of RTT values we have measured over the last interval
    this->last_snd_max = state->snd_max;
    this->last_snd_una = state->snd_una;
    this->last_ack_cnt = state->ack_cnt;
    this->last_bytes_lost = this->bytesLost;
    this->last_bytes_delivered = this->bytesDelivered;  
    this->lastIntervalTime = simTime();
}

// RayNet method: Called after simulation completion? Unsure how this differs from reset()
void Orca::cleanup()
{
    if (debug) cout << "\t" << stringId << " cleanUp()" << endl;
}

ObsType Orca::getRLState(){
    if (debug) cout << "\t" << stringId << " getRLState()" << endl;
    // Deprecated, remove this later
}

RewardType Orca::getReward(){
    if (debug) cout << "\t" << stringId << " getReward()" << endl;
    // Deprecated, remove this later
}

bool Orca::getDone() {
    return done;
    // Deprecated, remove this later (done is just set and checked directly)
}

// MARK: Cubic Methods
// ============================================================================
// These are copied directly from cubic.
// and a couple lines were added for tracking some extra information for Orca to use.
// ============================================================================

// Called upon an ACK. Store the RTT info for averaging at the end of the interval.
void Orca::rttMeasurementComplete(simtime_t tSent, simtime_t tAcked) {
    TcpCubic::rttMeasurementComplete(tSent, tAcked);
    double packetRTT = (tAcked-tSent).dbl();
    this->orcaDelaySum += packetRTT;
    this->rttReportCount += 1;

    
    this->orcaMinDelay = std::min(this->orcaMinDelay, packetRTT);
}

// Override to track bytes delivered
void Orca::receivedDataAck(uint32_t firstSeqAcked) {
    TcpCubic::receivedDataAck(firstSeqAcked);
    
    // this->bytesDelivered += state->snd_mss; // Number of bytes sent so far this interval
}
