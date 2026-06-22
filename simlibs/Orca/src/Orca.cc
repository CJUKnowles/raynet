#include "RLInterface.h"
#include "omnetpp/ccomponent.h"
#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/TcpPacedConnection.h"
#include "transportlayer/tcp/TcpPaced.h"
#include "transportlayer/tcp/flavours/TcpCubic.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <ostream>
#include "Orca.h"
#include "typedefs.h"
#include <inet/common/INETDefs.h>

using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(Orca); // Lets omnet see and use this class

Orca::Orca():
    TcpCubic(), RLInterface() {
    cout << "\tOrca: Constructor called!";
}

Orca::~Orca() {
    if (debug) cout << "\tOrca: Destructor method called. Goodbye." << endl;
    getSimulation()->getSystemModule()->unsubscribe(stringId.c_str(), (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("performAction", (cListener*) this);
}

// Called during sim initialization
void Orca::initialize() {
    if (debug) cout << "\tOrca initialize()" << endl;
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
        this->lastIntervalTime = simTime();

        // Set the RayNet ID of this agent and register with the Broker.
        RLInterface::registerRLAgent("Orca");

        // Finally, schedule this agent's first fixed-length RL step.
        RLInterface::scheduleNextStep(this->fixedIntervalDuration);
    }
}

// Return the raw transport metrics used by the original Orca environment wrapper.
std::optional<ObsType> Orca::computeObservation(){
    if (debug) cout << "\t" << stringId << " computeObservation()" << endl; 

    // Collect interval-level delivery and loss metrics.
    double lastIntervalDuration = (simTime() - this->lastIntervalTime).dbl();
    TcpPacedConnection* pacedConnection = dynamic_cast<TcpPacedConnection*>(conn);
    this->bytesDelivered = pacedConnection->getDelivered();
    this->bytesLost = pacedConnection->getTotalDetectedLostBytes() + this->rtoLostBytes;
    this->delta_bytes_delivered = this->bytesDelivered - this->last_bytes_delivered;
    this->delta_bytes_lost = this->bytesLost - this->last_bytes_lost;
    this->orcaThroughput = lastIntervalDuration > 0.0 ? this->delta_bytes_delivered / lastIntervalDuration : 0.0;
    this->lossRate = lastIntervalDuration > 0.0 ? this->delta_bytes_lost / lastIntervalDuration : 0.0;
    this->delta_snd_max = state->snd_max - this->last_snd_max;
    this->delta_snd_una = state->snd_una - this->last_snd_una;
    this->delta_ack_cnt = state->ack_cnt - this->last_ack_cnt;

    // Convert current TCP values to the units used by the original Orca server.
    double averageDelayUs = this->rttReportCount > 0 ? (this->orcaDelaySum / this->rttReportCount) * 1e6 : 0.0;
    double averageDelayMs = averageDelayUs / 1000.0;
    double pacingRate = pacedConnection->intersendingTime.dbl() > 0.0 ? state->snd_mss / pacedConnection->intersendingTime.dbl() : 0.0;
    double cwndPackets = state->snd_mss > 0 ? state->snd_cwnd / (double)state->snd_mss : 0.0;
    double ssthreshPackets = state->snd_mss > 0 ? state->ssthresh / (double)state->snd_mss : 0.0;
    double packetsOut = state->snd_mss > 0 ? pacedConnection->getBytesInFlight() / (double)state->snd_mss : 0.0;
    double retransOut = state->snd_mss > 0 ? this->retransmissionRate * lastIntervalDuration / state->snd_mss : 0.0;
    double srttUs = state->srtt.dbl() * 1e6 * 8.0;
    double minRttUs = this->orcaMinDelay * 1e6;
    double minRttMs = minRttUs / 1000.0;

    // Permanently hand congestion-window control to Orca after Cubic exits initial slow start.
    this->slowStartPassed = this->slowStartPassed || state->snd_cwnd > state->ssthresh;

    // Schedule the next raw metric collection interval.
    scheduleNextStep(this->fixedIntervalDuration);

    // Match the original Orca server: do not expose a learner state until the
    // transport reports an RTT-bearing interval.
    if (this->rttReportCount == 0) {
        this->last_bytes_delivered = this->bytesDelivered;
        this->last_snd_max = state->snd_max;
        this->last_snd_una = state->snd_una;
        this->last_ack_cnt = state->ack_cnt;
        return std::nullopt;
    }

    if (this->takeActions && !this->slowStartPassed) {
        uint32_t boostedCwnd = ceil(((double)state->snd_cwnd) * 1.1);
        if (debug) cout << "\t\tSlow-start boost cwnd from " << state->snd_cwnd << " to " << boostedCwnd << "(1.1x)" << endl;
        applyCwnd(boostedCwnd);
        this->last_bytes_lost = this->bytesLost;
        this->last_bytes_delivered = this->bytesDelivered;
        this->last_snd_max = state->snd_max;
        this->last_snd_una = state->snd_una;
        this->last_ack_cnt = state->ack_cnt;
        this->orcaThroughput=0.0;
        this->orcaDelaySum=0.0;
        this->orcaACKTotal=0.0;
        this->rttReportCount=0;
        this->lastIntervalTime = simTime();
        return std::nullopt;
    }

    // Print the collected raw metrics when debug output is enabled.
    if(debug) {
        cout << "-" << endl;
        cout << "\taverageDelayMs: " << averageDelayMs << endl;
        cout << "\taverageDelayUs: " << averageDelayUs << endl;
        cout << "\tthroughput: " << this->orcaThroughput << endl;
        cout << "\trttSamples: " << this->rttReportCount << endl;
        cout << "\tintervalDuration: " << lastIntervalDuration << endl;
        cout << "\tcwndPackets: " << cwndPackets << endl;
        cout << "\tpacingRate: " << pacingRate << endl;
        cout << "\tlossRate: " << this->lossRate << endl;
        cout << "\tsrttUs: " << srttUs << endl;
        cout << "\tminRttMs: " << minRttMs << endl;
        cout << "\tminRttUs: " << minRttUs << endl;
    } 

    // Emit throughput for OMNeT++ result recording.
    owner->emit(throughputSignal, this->orcaThroughput);

    ObsType observation({
        averageDelayMs,
        this->orcaThroughput,
        this->rttReportCount,
        lastIntervalDuration,
        50.0,
        cwndPackets,
        pacingRate,
        this->lossRate,
        srttUs,
        ssthreshPackets,
        packetsOut,
        retransOut,
        packetsOut,
        state->snd_mss,
        minRttMs,
        averageDelayUs,
        minRttUs
    });

    // Original Orca clears throughput and loss accounting when a valid
    // RTT-bearing state is handed to the learner.
    this->last_bytes_lost = this->bytesLost;
    this->last_bytes_delivered = this->bytesDelivered;

    return observation;
}

RewardType Orca::computeReward(){
    return -1.0;
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void Orca::decisionMade(ActionType action) {
    if (debug) cout << "\t" << stringId << " decisionMade()" << endl;

    // Olympus sends the target congestion window in packets. RayNet converts
    // the packet count to INET's byte-based cwnd and enforces one MSS.
    double requestedPackets = std::max((double)action, 1.0);
    double requestedBytes = std::ceil(requestedPackets * (double)state->snd_mss);
    double requestedCwnd = std::max(requestedBytes, (double)state->snd_mss);
    uint32_t newCwnd = (uint32_t)requestedCwnd;
    double multiplier = (double)newCwnd / (double)state->snd_cwnd;

    if (this->takeActions) {
        cout << "\t\tChanging cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x)" << endl;
        applyCwnd(newCwnd);
        owner->emit(actionSignal, multiplier); // Emit action for plotting
    } else {
        cout << "\t\t" << "NOT CHANGING cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x) NOT CHANGING !!!!!!" << endl;
    }
}

void Orca::applyCwnd(uint32_t newCwnd) {
    state->snd_cwnd = newCwnd;

    // Update pacing rate with the Cubic-compatible pacing behavior used by this INET model.
    if(state->snd_cwnd > 0) {
        double paceFactor;
        if (state->snd_cwnd < state->ssthresh/2) {
            paceFactor = 2;
        }
        else{
            paceFactor = 1.2;
        }
        TcpPacedConnection* pacedConnection = dynamic_cast<TcpPacedConnection*>(conn);
        uint32_t maxWindow = std::max(state->snd_cwnd, pacedConnection->getBytesInFlight());
        double pace = state->srtt.dbl()/(((double) (maxWindow) / (double)state->snd_mss) * paceFactor);
        pacedConnection->changeIntersendingTime(pace);
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
    this->lastIntervalTime = simTime();
}

void Orca::processRexmitTimer(TcpEventCode &event) {
    TcpPacedConnection* pacedConnection = dynamic_cast<TcpPacedConnection*>(conn);
    uint64_t bytesInFlight = pacedConnection->getBytesInFlight();

    TcpCubic::processRexmitTimer(event);

    if (event != TCP_E_ABORT) {
        this->rtoLostBytes += bytesInFlight;
    }
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

void Orca::receivedDuplicateAck() {
    TcpCubic::receivedDuplicateAck();
}
