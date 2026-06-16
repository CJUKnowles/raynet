#include "omnetpp/ccomponent.h"
#include "omnetpp/cobject.h"
#include "omnetpp/simtime.h"
#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/TcpPacedConnection.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include "Astraea.h"
#include "typedefs.h"
#include <inet/common/INETDefs.h>

using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(Astraea); // Lets OMNeT++ see and use this class.

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

    std::string makeAstraeaAgentId(cComponent *owner)
    {
        cModule *tcpModule = dynamic_cast<cModule *>(owner);
        cModule *hostModule = tcpModule ? tcpModule->getParentModule() : nullptr;
        int hostIndex = hostModule ? hostModule->getIndex() : -1;

        if (hostIndex == 0) {
            return "Astraea";
        }
        if (hostIndex > 0) {
            return "Astraea" + std::to_string(hostIndex);
        }
        return "Astraea_" + sanitizeAgentId(owner ? owner->getFullPath() : "unknown");
    }
}

Astraea::Astraea():
    TcpPacedNoCC(), RLInterface() {
    if (debug) cout << "\t" << stringId << ": Constructor called!";
}

Astraea::~Astraea() {
    if (debug) cout << "\t" << stringId << ": Destructor method called. Goodbye.";
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
        RLInterface::setStringId(makeAstraeaAgentId(owner));
        cObject* simtime = new cSimTime(0);
        owner->emit(this->registerSig, stringId.c_str(), simtime);

        // Finally, schedule this agent's first fixed-length RL step.
        scheduleNextStep(this->fixedIntervalDuration);
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
    double throughputBytesPerSecond = deltaBytesDelivered / lastIntervalDuration;
    this->astraeaThroughput = throughputBytesPerSecond;
    this->astraeaMaxThroughput = std::max(this->astraeaMaxThroughput, throughputBytesPerSecond);
    this->astraeaLossRate = lastIntervalDuration > 0.0 ? deltaBytesLost / lastIntervalDuration : 0.0;
    double averageRttUs = this->rttReportCount > 0 ? (this->astraeaDelaySum / this->rttReportCount) * 1e6 : 0.0;
    double minRttUs = this->rttReportCount > 0 ? this->astraeaMinDelay * 1e6 : 0.0;
    double srttUs = state->srtt.dbl() * 1e6 * 8.0;
    double cwndPackets = state->snd_mss > 0 ? state->snd_cwnd / (double)state->snd_mss : 0.0;
    double packetsOut = state->snd_mss > 0 ? pacedConnection->getBytesInFlight() / (double)state->snd_mss : 0.0;
    double pacingRateBytesPerSecond = pacedConnection->intersendingTime.dbl() > 0.0 ? state->snd_mss / pacedConnection->intersendingTime.dbl() : 0.0;
    double retransOutPackets = state->snd_mss > 0 ? deltaBytesLost / state->snd_mss : 0.0;

    // Print the collected raw metrics when debug output is enabled.
    if(debug) {
        cout << "-" << endl;
        cout << "\t" << stringId << " step #" << this->RLStepsTaken << ":" << endl;
        cout << "\t\tavg_thr: " << throughputBytesPerSecond << endl;
        cout << "\t\tmax_tput: " << this->astraeaMaxThroughput << endl;
        cout << "\t\tavg_urtt: " << averageRttUs << endl;
        cout << "\t\tmin_rtt: " << minRttUs << endl;
        cout << "\t\tsrtt_us: " << srttUs << endl;
        cout << "\t\tcwnd: " << cwndPackets << endl;
        cout << "\t\tloss_ratio: " << this->astraeaLossRate << endl;
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
        throughputBytesPerSecond,
        this->astraeaMaxThroughput,
        averageRttUs,
        minRttUs,
        srttUs,
        cwndPackets,
        this->astraeaLossRate,
        packetsOut,
        pacingRateBytesPerSecond,
        retransOutPackets,
    });
}

RewardType Astraea::computeReward(){
    return 0.0;
}

// RayNet method: make a decision based on the policy by altering snd_cwnd.
void Astraea::decisionMade(ActionType action) {
    if (debug) cout << "\t" << stringId << ": decisionMade()" << endl;

    // Calculate the new cwnd using the original Astraea action transform.
    double scaledCwnd;
    if (action >= 0) {
        scaledCwnd = (double) state->snd_cwnd * (1.0 + this->actionControlCoeff * action);
        scaledCwnd = std::ceil(scaledCwnd);
    } else {
        scaledCwnd = (double) state->snd_cwnd / (1.0 - this->actionControlCoeff * action);
        scaledCwnd = std::floor(scaledCwnd);
    }
    uint32_t newCwnd = (uint32_t) scaledCwnd;
    newCwnd = max(newCwnd, state->snd_mss);
    double multiplier = (double) newCwnd / state->snd_cwnd;

    // Attempt to change cwnd and pacing rate.
    if (this->takeActions && this->rttReportCount > 0) {
        if (debug) cout << "\t\tChanging cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x)" << endl;
        state->snd_cwnd = newCwnd;

        // Pace one max(cwnd, bytes-in-flight) window per RTT, matching original Astraea.
        if(state->snd_cwnd > 0) {
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
        this->deliveryRateSampleSum += sample.m_deliveryRate;
        this->deliveryRateSampleCount += 1;
    }
}
