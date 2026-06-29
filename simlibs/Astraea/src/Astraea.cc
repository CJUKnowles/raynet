#include "omnetpp/ccomponent.h"
#include "omnetpp/cobject.h"
#include "omnetpp/simtime.h"
#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/TcpPacedConnection.h"
#include "inet/transportlayer/tcp/TcpSackRexmitQueue.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include "Astraea.h"
#include "typedefs.h"
#include <inet/common/INETDefs.h>

using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(Astraea); // Lets OMNeT++ see and use this class.

Astraea::Astraea():
    TcpPacedNoCC(), RLInterface() {
    if (debug) cout << "\t" << stringId << ": Constructor called!";
}

Astraea::~Astraea() {
    if (debug) cout << "\t" << stringId << ": Destructor method called. Goodbye." << endl;
    getSimulation()->getSystemModule()->unsubscribe(stringId.c_str(), (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("performAction", (cListener*) this);
}

// Called during sim initialization.
void Astraea::initialize() {
    if (debug) cout << "\t" << stringId << " initialize()" << endl;
    this->fixedIntervalDuration = this->conn->getTcpMain()->par("fixedIntervalDuration");
    this->actionControlCoeff = this->conn->getTcpMain()->par("actionControlCoeff");
    debug = this->conn->getTcpMain()->par("printDebugMessages");
    takeActions = this->conn->getTcpMain()->par("takeActions");

    // Provide the RLInterface with a cComponent API for signals.
    setOwner((cComponent*) conn->getTcpMain());

    // Initialize parent classes.
    RLInterface::initialise();
    TcpPacedNoCC::initialize();

    // Signals for OMNeT++ result recording.
    throughputSignal = conn->registerSignal("throughput");
    srttSignal = conn->registerSignal("srtt");
    cwndSignal = conn->registerSignal("cwnd");
    intervalDurationSignal = conn->registerSignal("intervalDuration");
    actionSignal = conn->registerSignal("action");
}

// INET method, called after the TCP connection is established.
void Astraea::established(bool active) {
    if (debug) cout << "\t" << stringId << ": established()" << endl;
    TcpPacedNoCC::established(active);

    // Only register this as an RL agent if this is a client.
    if (active) {
        this->isActive = active;
        this->lastIntervalTime = simTime();

        // Set the RayNet ID of this agent and register with the Broker.
        RLInterface::registerRLAgent("Astraea");

        // Finally, schedule this agent's first fixed-length RL step.
        scheduleNextStep(this->fixedIntervalDuration);

        dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(.01);
    }
}

void Astraea::connectionClosed() {
    if (debug) cout << "\t" << stringId << ": connectionClosed()" << endl;
    TcpPacedNoCC::connectionClosed();

    done = true;
    if (isActive) {
        RLInterface::terminate();
    }
}

// Return the raw transport metrics used by the original Astraea environment wrapper.
std::optional<ObsType> Astraea::computeObservation(){
    if (debug) cout << "\t" << stringId << " computeObservation()" << endl;

    // Collect interval-level delivery, loss, and delay metrics.
    double lastIntervalDuration = (simTime() - this->lastIntervalTime).dbl();
    if (lastIntervalDuration <= 0.0) {
        lastIntervalDuration = this->fixedIntervalDuration;
    }
    TcpPacedConnection* pacedConnection = dynamic_cast<TcpPacedConnection*>(conn);
    this->bytesDelivered = pacedConnection->getDelivered();
    this->bytesLost = pacedConnection->getTotalDetectedLostBytes();
    double deltaBytesDelivered = this->bytesDelivered - this->last_bytes_delivered;
    double deltaBytesLost = this->bytesLost - this->last_bytes_lost;

    // Convert current TCP values to the units used by the original Astraea helper.
    double intervalThroughputBytesPerSecond = 0.0;
    if (lastIntervalDuration > 0.0) {
        intervalThroughputBytesPerSecond = deltaBytesDelivered / lastIntervalDuration;
    }
    double unweightedRateSampleThroughputBytesPerSecond = 0.0;
    if (this->deliveryRateSampleCount > 0) {
        unweightedRateSampleThroughputBytesPerSecond =
            this->deliveryRateSampleSum / this->deliveryRateSampleCount;
    }
    double weightedRateSampleThroughputBytesPerSecond = 0.0;
    if (this->deliveryRateSampleIntervalSum > 0.0) {
        weightedRateSampleThroughputBytesPerSecond =
            this->deliveryRateSampleWeightedSum / this->deliveryRateSampleIntervalSum;
    }
    double throughputBytesPerSecond = weightedRateSampleThroughputBytesPerSecond;
    this->astraeaThroughput = throughputBytesPerSecond;
    this->astraeaMaxThroughput = std::max(this->astraeaMaxThroughput, throughputBytesPerSecond);
    this->astraeaLossRate = lastIntervalDuration > 0.0 ? deltaBytesLost / lastIntervalDuration : 0.0;
    double averageRttUs = this->rttReportCount > 0 ? (this->astraeaDelaySum / this->rttReportCount) * 1e6 : 0.0;
    double minRttUs = this->astraeaMinDelay * 1e6;
    double srttUs = state->srtt.dbl() * 1e6 * 8.0;
    double cwndPackets = state->snd_mss > 0 ? state->snd_cwnd / (double)state->snd_mss : 0.0;
    double packetsOut = state->snd_mss > 0 ? pacedConnection->getBytesInFlight() / (double)state->snd_mss : 0.0;
    double pacingRateBytesPerSecond = pacedConnection->intersendingTime.dbl() > 0.0 ? state->snd_mss / pacedConnection->intersendingTime.dbl() : 0.0;
    double retransOutPackets = 0.0;
    if (state->snd_mss > 0 && pacedConnection->getRexmitQueue() != nullptr) {
        retransOutPackets = pacedConnection->getRexmitQueue()->getTotalRetransmitted() / (double)state->snd_mss;
    }

    // Print the collected raw metrics when debug output is enabled.
    if(debug) {
        cout << "-" << endl;
        cout << "\t" << stringId << " step #" << this->RLStepsTaken << ":" << endl;
        cout << "\t\tavg_thr: " << throughputBytesPerSecond << endl;
        cout << "\t\tavg_thr_INTERVAL_DEBUG: " << intervalThroughputBytesPerSecond << endl;
        cout << "\t\tavg_thr_OLD_DEBUG: " << unweightedRateSampleThroughputBytesPerSecond << endl;
        cout << "\t\tavg_thr_SAMPLE_INTERVAL_DEBUG: " << this->deliveryRateSampleIntervalSum << endl;
        cout << "\t\tavg_thr_SAMPLE_COUNT_DEBUG: " << this->deliveryRateSampleCount << endl;
        cout << "\t\tmax_tput: " << this->astraeaMaxThroughput << endl;
        cout << "\t\tavg_urtt: " << averageRttUs << endl;
        cout << "\t\tmin_rtt: " << minRttUs << endl;
        cout << "\t\tsrtt_us: " << srttUs << endl;
        cout << "\t\tcwnd: " << cwndPackets << endl;
        cout << "\t\tloss_rate: " << this->astraeaLossRate << endl;
        cout << "\t\tpackets_out: " << packetsOut << endl;
        cout << "\t\tpacing_rate: " << pacingRateBytesPerSecond << endl;
        cout << "\t\tretrans_out: " << retransOutPackets << endl;
        cout << "-" << endl;
    }

    // Emit common metrics for OMNeT++ result recording.
    conn->emit(throughputSignal, throughputBytesPerSecond);
    conn->emit(srttSignal, state->srtt);
    conn->emit(cwndSignal, state->snd_cwnd);

    // Python owns episode truncation and reward/global-state calculation.
    RLStepsTaken++;
    this->pendingIntervalReset = true;
    scheduleNextStep(this->fixedIntervalDuration);

    return ObsType({
        {"avg_thr", throughputBytesPerSecond},
        {"avg_thr_INTERVAL_DEBUG", intervalThroughputBytesPerSecond},
        {"avg_thr_OLD_DEBUG", unweightedRateSampleThroughputBytesPerSecond},
        {"avg_thr_SAMPLE_INTERVAL_DEBUG", this->deliveryRateSampleIntervalSum},
        {"avg_thr_SAMPLE_COUNT_DEBUG", (double)this->deliveryRateSampleCount},
        // {"throughput", throughputBytesPerSecond},
        // {"max_tput", this->astraeaMaxThroughput},
        {"avg_urtt", averageRttUs},
        // {"delay_us", averageRttUs},
        {"min_rtt", minRttUs},
        // {"min_rtt_us", minRttUs},
        {"srtt_us", srttUs},
        {"cwnd", cwndPackets},
        // {"loss_rate", this->astraeaLossRate},
        {"packets_out", packetsOut},
        {"pacing_rate", pacingRateBytesPerSecond},
        {"retrans_out", retransOutPackets},
    });
}

RewardType Astraea::computeReward(){
    return 0.0;
}

// RayNet method: make a decision based on the policy by altering snd_cwnd.
void Astraea::decisionMade(ActionType action) {
    if (debug) cout << "\t" << stringId << ": decisionMade()" << endl;

    // Olympus sends the target congestion window in packets. RayNet converts
    // the packet count to INET's byte-based cwnd and enforces one MSS.
    double requestedPackets = std::max((double)action, 1.0);
    double requestedBytes = std::ceil(requestedPackets * (double)state->snd_mss);
    double requestedCwnd = std::max(requestedBytes, (double)state->snd_mss);
    uint32_t newCwnd = (uint32_t)requestedCwnd;
    double multiplier = (double) newCwnd / (double) state->snd_cwnd;

    // Attempt to change cwnd and pacing rate.
    if (this->takeActions) {
        if (debug) cout << "\t\tChanging cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x)" << endl;
        state->snd_cwnd = newCwnd;

        // Pace one max(cwnd, bytes-in-flight) window per RTT, matching original Astraea.
        if(state->snd_cwnd > 0 && state->srtt > SIMTIME_ZERO) {
            uint32_t maxWindow = std::max(state->snd_cwnd, dynamic_cast<TcpPacedConnection*>(conn)->getBytesInFlight());
            double pace = state->srtt.dbl() / ((double) maxWindow / (double) state->snd_mss);
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

void Astraea::resetStepVariables()
{
    if (debug) cout << "\t" << stringId << ": resetStepVariables()" << endl;
    if (!this->pendingIntervalReset) {
        return;
    }

    // Commit the interval boundary and begin collecting the next interval.
    this->deliveryRateSampleSum = 0.0;
    this->deliveryRateSampleWeightedSum = 0.0;
    this->deliveryRateSampleIntervalSum = 0.0;
    this->deliveryRateSampleCount = 0;
    this->astraeaDelaySum = 0.0;
    this->rttReportCount = 0;
    this->last_bytes_lost = this->bytesLost;
    this->last_bytes_delivered = this->bytesDelivered;
    this->lastIntervalTime = simTime();
    this->pendingIntervalReset = false;
}

bool Astraea::getDone() {
    return done;
}

void Astraea::cleanup()
{
    if (debug) cout << "\t" << stringId << ": cleanUp()" << endl;
}

ObsType Astraea::getRLState(){
    if (debug) cout << "\t" << stringId << ": getRLState()" << endl;
    return ObsType();
}

RewardType Astraea::getReward(){
    if (debug) cout << "\t" << stringId << ": getReward()" << endl;
    return 0.0;
}

void Astraea::rttMeasurementComplete(simtime_t tSent, simtime_t tAcked) {
    TcpPacedNoCC::rttMeasurementComplete(tSent, tAcked);
    double packetRTT = (tAcked - tSent).dbl();
    this->astraeaDelaySum += packetRTT;
    this->rttReportCount += 1;
    if (this->astraeaMinDelay == 0.0) {
        this->astraeaMinDelay = packetRTT;
    } else {
        this->astraeaMinDelay = std::min(this->astraeaMinDelay, packetRTT);
    }
}

void Astraea::receivedDataAck(uint32_t firstSeqAcked) {
    TcpPacedNoCC::receivedDataAck(firstSeqAcked);
    recordDeliveryRateSample();
}

void Astraea::receivedDuplicateAck() {
    TcpPacedNoCC::receivedDuplicateAck();
    recordDeliveryRateSample();
}

void Astraea::recordDeliveryRateSample() {
    TcpPacedConnection *pacedConnection = dynamic_cast<TcpPacedConnection *>(conn);
    TcpPacedConnection::RateSample sample = pacedConnection->getRateSample();
    if (sample.m_interval > SIMTIME_ZERO) {
        double sampleInterval = sample.m_interval.dbl();
        this->deliveryRateSampleSum += sample.m_deliveryRate;
        this->deliveryRateSampleWeightedSum +=
            (double)sample.m_deliveryRate * sampleInterval;
        this->deliveryRateSampleIntervalSum += sampleInterval;
        this->deliveryRateSampleCount += 1;
    }
}
