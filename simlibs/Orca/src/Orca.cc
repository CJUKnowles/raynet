#include "RLInterface.h"
#include "omnetpp/ccomponent.h"
#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/TcpPacedConnection.h"
#include "transportlayer/tcp/TcpPaced.h"
#include "transportlayer/tcp/flavours/TcpCubic.h"
#include <algorithm>
#include <numeric>
#include <ostream>
#ifdef ORCA
#include "Orca.h"
#include "typedefs.h"
#include <inet/common/INETDefs.h>

using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(Orca); // Lets omnet see and use this class

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
    this->rewardDelayForgiveness = this->conn->getTcpMain()->par("rewardDelayForgiveness");
    this->rewardLossMultiplier = this->conn->getTcpMain()->par("rewardLossMultiplier");
    this->fixedIntervals = this->conn->getTcpMain()->par("fixedIntervals");
    this->fixedIntervalDuration = this->conn->getTcpMain()->par("fixedIntervalDuration");
    this->maxRLSteps = this->conn->getTcpMain()->par("maxRLSteps");
    this->takeActions = this->conn->getTcpMain()->par("takeActions");
    this->debug = this->conn->getTcpMain()->par("printDebugMessages");

    // provide the RLInterface with a cComponent API (for signals)
    RLInterface::setOwner((cComponent*) conn->getTcpMain());
    
    // Initalize parent classes
    RLInterface::initialise();
    TcpCubic::initialize();

    // Register metric signals (for plotting)
    throughputSignal = owner->registerSignal("throughput");
    actionSignal = owner->registerSignal("action");
    // Suscribe to important signals:
    TcpPacedConnection* pacedConn = dynamic_cast<TcpPacedConnection*>(conn);
    pacedConn->subscribe(pacedConn->retransmissionRateSignal, (cListener*) this);
}

// OMNet Method? Called after component initialization is complete?
void Orca::established(bool active) {
    if (debug) cout << "\tOrca: established()" << endl;
    TcpCubic::established(active);
    
    // Only initialize the RL agent if this is a client
    if (active) {
        this->isActive = active;

        // Set the RayNet ID of this agent and register with the Broker
        RLInterface::setStringId("Orca");
        cObject* simtime = new cSimTime(0); // Used to contain initial step length, now deprecated.
        owner->emit(this->registerSig, stringId.c_str(), simtime); 

        // Finally, schedule this agent's first RL step (first step uses fixedIntervalDuration, even if fixedIntervals==false)
        RLInterface::scheduleNextStep(this->fixedIntervalDuration);
    }
}

// Return an observation to the broker, based on the current state
std::optional<ObsType> Orca::computeObservation(){
    if (debug) cout << "\tOrca: computeObservation()" << endl; 
    double lastIntervalDuration = (simTime() - this->lastIntervalTime).dbl();

    this->delta_snd_max = state->snd_max - this->last_snd_max;
    this->delta_snd_una = state->snd_una - this->last_snd_una;
    this->delta_ack_cnt = state->ack_cnt - this->last_ack_cnt;
    
    // Initialize empty obs to populate as we go
    double obs[7] = {0,0,0,0,0,0,};

    // Throughput: How many bytes were DELIVERED this interval (basically goodput?)
    this->orcaThroughput = this->bytesDelivered / lastIntervalDuration;
    this->orcaMaxThroughput = std::max(this->orcaMaxThroughput, this->orcaThroughput);
    if (this->orcaMaxThroughput == 0.0) {
        obs[0] = obs[1] = obs[2] = 0.0;
    } else {
        // Throughput
        obs[0] = this->orcaThroughput / this->orcaMaxThroughput;

        // Pacerate
        this->orcaPaceRate = (1.0/dynamic_cast<TcpPacedConnection*>(conn)->intersendingTime.dbl()) * (double) state->snd_mss;
        obs[1] = std::min(10.0, this->orcaPaceRate / this->orcaMaxThroughput); // Clamp to 10 as per paper (paceRate before first action is massive)

        // Lossrate: What percentage of bytes sent this interval were retransmissions
        this->orcaLossRate = 0.0;
        if (this->retransmissionRate > 0.0) {  // Avoid division by 0
            double transmissionRate = delta_snd_max/lastIntervalDuration; // How many non-retransmits occurred this interval in bytes/s
            this->orcaLossRate = this->retransmissionRate / (this->retransmissionRate + transmissionRate); // What percentage of interval's sent data was retransmissions
        }
        obs[2] = this->retransmissionRate / this->orcaMaxThroughput;
    }

    // ACKed: How many bytes were ACKed this interval
    this->orcaACKTotal= this->bytesDelivered/(double)state->snd_mss;
    obs[3] = this->orcaACKTotal /  state->snd_cwnd;
    
    // Interval Duration: How many seconds passed since last interval
    obs[4] = lastIntervalDuration;

    // Delay: Tracked in overridden method above. Only update the minimum if delay reports were received this interval.
    if (this->rttReportCount == 0 || state->srtt.dbl() == 0.0) {
        // No RTT reports were received this interval. Set all RTT related values to 0 to represent lack of data.
        this->orcaDelay = 0.0; // Deprecated?
        this->orcaDelayMetric = 0.0; // Delay is unobserved, not 0. Do not report as optimal.
        obs[5] = 0.0; // SRTT
        obs[6] = 0.0; // Delay metric
    } else {
        // RTT reports were received this interval. Compute the average and potentially update the minimum.
        this->orcaDelay = this->orcaDelaySum/this->rttReportCount;

        // Compute the delay metric (0.0 is poor, 1.0 is optimal. Report as optimal if within the forgiveness window.)
        if (state->srtt > this->orcaMinDelay * this->rewardDelayForgiveness) {                                             
            this->orcaDelayMetric = this->orcaMinDelay * this->rewardDelayForgiveness / state->srtt;
        } else {
            this->orcaDelayMetric = 1.0;
        }
        obs[5] = this->orcaMinDelay / std::max(state->srtt.dbl(), this->orcaMinDelay); // sRTT is lower than it should be at startup. The max prevents this obs from being > 1 when that happens.
        obs[6] = this->orcaDelayMetric;
    }
    

    // Update step count, and check if the step limit has been reached
    RLStepsTaken++;
    if (RLStepsTaken >= this->maxRLSteps) {
        done = true;
    } else {
        // Finally, schedule the next step (will be automatically cancelled if done)
        scheduleNextStep(this->fixedIntervals ? this->fixedIntervalDuration : state->srtt.dbl());
    }

    // Debug prints
    if(debug) {
        cout << "-" << endl;
        cout << "! STEP: " << RLStepsTaken << " !" << endl;
        cout << "\tState:" << endl;
            cout << "\t\tsnd_una: " << state->snd_una << endl;
            cout << "\t\tdelta_snd_una: " << delta_snd_una << endl;
            cout << "\t\tcwnd: " << state->snd_cwnd << endl;
            cout << "\t\tsnd_max: " << state->snd_max << endl;
            cout << "\t\tSRTT: " << state->srtt << endl;
            cout << "\t\tack_cnt: " << state->ack_cnt << endl;
            cout << "\t\tdelta_ack_cnt: " << this->delta_ack_cnt << endl;
            cout << "\t\tACKTotal: " << this->orcaACKTotal << endl;
            cout << "\t\tpacketsDelivered" << this->bytesDelivered/state->snd_mss << endl;
            cout << "\t\trttReportCount: " << this->rttReportCount << endl;
            cout << "\t\tDone: " << done << endl;
        cout << "\tObservations:" << endl;
            cout << "\t\tThroughput: " << obs[0] << endl;
            cout << "\t\tPacerate: " << obs[1] << endl;
            cout << "\t\tLossRate: " << obs[2] << endl;
            cout << "\t\tAcksCount: " << obs[3] << endl;
            cout << "\t\tIntervalDuration: " << obs[4] << endl;
            cout << "\t\tRaw SRTT: " << obs[5] << endl;
            cout << "\t\tSRTT Metric: " << obs[6] << endl;
    } 

    // Plotting metrics
    owner->emit(throughputSignal, this->orcaThroughput);

    return ObsType{
            obs[0],     // Normalized throughput
            obs[1],       // Normalized pacerate
            obs[2], // Normalized lossrate
            obs[3],              // Normalized ACKs count (maybe use tcp_cwnd? ask aiden)     
            obs[4],                               // Monitor interval duration
            obs[5],             // Normalized SRTT (delay)
            obs[6]                               // Normalized SRTT (possibly forgiven, if within the forgiveness window)
        };
}

RewardType Orca::computeReward(){
    if (debug) cout << "\tOrca: computeReward()" << endl;
    double reward;
    if (this->orcaMaxThroughput == 0.0) {
        reward = 0.0;
    } else {
        reward = (this->orcaThroughput-(this->rewardLossMultiplier*this->orcaLossRate))/this->orcaMaxThroughput*this->orcaDelayMetric;
    }
    if (debug) cout << "\t\tReward: " << reward << endl;
    return(reward);
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void Orca::decisionMade(ActionType action) {
    if (debug) cout << "\tOrca: decisionMade()" << endl;

    // Compute new cwnd from the given action (cwnd *= 2^action)
    double multiplier = std::pow(2.0, (double) action);
    uint32_t newCwnd = ceil(((double) state->snd_cwnd) * multiplier);
    newCwnd = max(newCwnd, state->snd_mss);
    
    // Attempt to change cwnd and pacing rate
    if (this->takeActions && this->rttReportCount > 0 && newCwnd <= 1000000) {
        if (debug) cout << "\t\tChanging cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x)" << endl;
        state->snd_cwnd = newCwnd;
        this->orcaPaceRate = ((double) state->snd_cwnd / (double) state->snd_mss) / state->srtt.dbl();  // Segments/s
        dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(1.0/orcaPaceRate); // 1/paceRate = intersendingtime
        owner->emit(actionSignal, multiplier); // Emit action for plotting
    } else {
        // Invalid. Skip this action entirely.
        if (debug) {
            cout << "\t\t" << "NOT CHANGING cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << multiplier << "x) NOT CHANGING !!!!!!" << endl;
        } 
    }

    if (debug) {
        cout << "\t\t" << (this->takeActions) << endl;
        // cout << "\t\t" << (this->first_slowstart_complete) << endl;
        cout << "\t\t" << (this->rttReportCount > 0) << endl;
        cout << "\t\t" << (newCwnd < 1000000) << endl;
        cout << "-" << endl;
    }
}


void Orca::resetStepVariables()
{
    if (debug) cout << "\t\tOrca: resetStepVariables()" << endl;
    this->orcaThroughput=0.0;    // The average delivery rate (throughput) over the last interval
    this->orcaLossRate=0.0;      // The average loss rate of packets over the last interval
    this->orcaDelay=0.0;         // The average delay of packets over the last interval
    this->orcaDelaySum=0.0;      // Sum of all RTT reports received over an interval
    this->orcaACKTotal=0.0;      // The number of valid acknowledgements over the last interval
    this->bytesDelivered=0.0;
    this->rttReportCount=0; // The number of RTT values we have measured over the last interval
    this->last_snd_max = state->snd_max;
    this->last_snd_una = state->snd_una;
    this->last_ack_cnt = state->ack_cnt;
    this->lastIntervalTime = simTime();
}

// RayNet method: Called after simulation completion? Unsure how this differs from reset()
void Orca::cleanup()
{
    if (debug) cout << "\tOrca: cleanUp()" << endl;
}

ObsType Orca::getRLState(){
    if (debug) cout << "\tOrca: getRLState()" << endl;
    // Deprecated, remove this later
}

RewardType Orca::getReward(){
    if (debug) cout << "\tOrca: getReward()" << endl;
    // Deprecated, remove this later
}

bool Orca::getDone() {
    return done;
    // Deprecated, remove this later (done is just set and checked directly)
}

// MARK: Cubic Methods
// ============================================================================
// These are copied directly from cubic. Any lines that alter the pacing rate were commented out,
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
    TcpTahoeRenoFamily::receivedDataAck(firstSeqAcked);
    // std::cout << "cwnd:::: " << state->snd_cwnd << endl;
    // std::cout << "snd_max::::" << state->snd_max << endl;
    // std::cout << "snd_una::::" << state->snd_una << endl;
    state->delay_min = state->srtt.inUnit(SIMTIME_US);
    // Check if recovery phase has ended
    if (state->sack_enabled && state->lossRecovery) {
        //dynamic_cast<PacedTcpConnection*>(conn)->changeIntersendingTime(0.000000001);
        // RFC 3517, page 7: "Once a TCP is in the loss recovery phase the following procedure MUST
        // be used for each arriving ACK:
        //
        // (A) An incoming cumulative ACK for a sequence number greater than
        // RecoveryPoint signals the end of loss recovery and the loss
        // recovery phase MUST be terminated.  Any information contained in
        // the scoreboard for sequence numbers greater than the new value of
        // HighACK SHOULD NOT be cleared when leaving the loss recovery
        // phase."
        if (seqGE(state->snd_una, state->recoveryPoint)) {
            EV_INFO << "Loss Recovery terminated.\n";
            state->snd_cwnd = state->ssthresh;
            state->lossRecovery = false;
        }
        else{
            dynamic_cast<TcpPacedConnection*>(conn)->doRetransmit();
            //conn->setPipe();
            //if (((int)state->snd_cwnd - (int)state->pipe) >= (int)state->snd_mss) // Note: Typecast needed to avoid prohibited transmissions
            //    dynamic_cast<TcpPacedConnection*>(conn)->sendDataDuringLossRecoveryPhase(state->snd_cwnd);
        }
        conn->emit(sndUnaSignal, state->snd_una);
        conn->emit(recoveryPointSignal, state->recoveryPoint);
    }

    if (state->snd_cwnd < state->ssthresh) {
        EV_INFO << "cwnd <= ssthresh: Slow Start: increasing cwnd by one SMSS bytes to ";

        // perform Slow Start. RFC 2581: "During slow start, a TCP increments cwnd
        // by at most SMSS bytes for each ACK received that acknowledges new data."
        state->snd_cwnd += state->snd_mss;
        conn->emit(cwndSignal, state->snd_cwnd);
        conn->emit(ssthreshSignal, state->ssthresh);

        EV_INFO << "cwnd=" << state->snd_cwnd << "\n";
    }
    else {

        updateCubicCwnd(1);

        if (state->cwnd_cnt >= state->cnt) {
            state->snd_cwnd += state->snd_mss;
            state->cwnd_cnt = 0;
        }
        else {
            state->cwnd_cnt++;
        }
        conn->emit(cwndSignal, state->snd_cwnd);
        conn->emit(ssthreshSignal, state->ssthresh);


        EV_INFO << "cwnd > ssthresh: Congestion Avoidance: increasing cwnd linearly, to " << state->snd_cwnd << "\n";
    }

    if(state->snd_cwnd > 0){
        double paceFactor;
        if (state->snd_cwnd < state->ssthresh/2) {
            paceFactor = 2;
        }
        else{
            paceFactor = 1.2;
        }
        uint32_t maxWindow = std::max(state->snd_cwnd, dynamic_cast<TcpPacedConnection*>(conn)->getBytesInFlight());
        // ORCA: Commented out this pacing rate change!
        //dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(state->srtt.dbl()/(((double) maxWindow/(double)state->snd_mss)* paceFactor));
    }

    sendData(false);

    conn->emit(cwndSegSignal, state->snd_cwnd / state->snd_mss);

    if(!this->first_slowstart_complete && state->snd_cwnd >= state->ssthresh) {
        this->first_slowstart_complete=true;
    }
    this->bytesDelivered += state->snd_mss; // Number of bytes sent so far this interval
}

void Orca::receivedDuplicateAck()
{
    //TcpTahoeRenoFamily::receivedDuplicateAck();
    state->delay_min = state->srtt.inUnit(SIMTIME_US);

    bool isHighRxtLost = dynamic_cast<TcpPacedConnection*>(conn)->checkIsLost(state->snd_una+state->snd_mss);
    bool rackLoss = dynamic_cast<TcpPacedConnection*>(conn)->checkRackLoss();
    if ((rackLoss && !state->lossRecovery) || state->dupacks == state->dupthresh || (isHighRxtLost && !state->lossRecovery)) {
        EV_INFO << "Reno on dupAcks == DUPTHRESH(=" << state->dupthresh << ": perform Fast Retransmit, and enter Fast Recovery:";

        if (state->sack_enabled) {
            // RFC 3517, page 6: "When a TCP sender receives the duplicate ACK corresponding to
            // DupThresh ACKs, the scoreboard MUST be updated with the new SACK
            // information (via Update ()).  If no previous loss event has occurred
            // on the connection or the cumulative acknowledgment point is beyond
            // the last value of RecoveryPoint, a loss recovery phase SHOULD be
            // initiated, per the fast retransmit algorithm outlined in [RFC2581].
            // The following steps MUST be taken:
            //
            // (1) RecoveryPoint = HighData
            //
            // When the TCP sender receives a cumulative ACK for this data octet
            // the loss recovery phase is terminated."

            // RFC 3517, page 8: "If an RTO occurs during loss recovery as specified in this document,
            // RecoveryPoint MUST be set to HighData.  Further, the new value of
            // RecoveryPoint MUST be preserved and the loss recovery algorithm
            // outlined in this document MUST be terminated.  In addition, a new
            // recovery phase (as described in section 5) MUST NOT be initiated
            // until HighACK is greater than or equal to the new value of
            // RecoveryPoint."
            if (state->recoveryPoint == 0 || seqGE(state->snd_una, state->recoveryPoint)) { // HighACK = snd_una
                state->recoveryPoint = state->snd_max; // HighData = snd_max
                dynamic_cast<TcpPacedConnection*>(conn)->setSackedHeadLost();
                dynamic_cast<TcpPacedConnection*>(conn)->updateInFlight();
                state->lossRecovery = true;

                recalculateSlowStartThreshold();
                state->snd_cwnd = state->ssthresh + (3*state->snd_mss); // 20051129 (1)
                EV_DETAIL << " recoveryPoint=" << state->recoveryPoint;

                dynamic_cast<TcpPacedConnection*>(conn)->doRetransmit();
            }
        }
        // RFC 2581, page 5:
        // "After the fast retransmit algorithm sends what appears to be the
        // missing segment, the "fast recovery" algorithm governs the
        // transmission of new data until a non-duplicate ACK arrives.
        // (...) the TCP sender can continue to transmit new
        // segments (although transmission must continue using a reduced cwnd)."

        // enter Fast Recovery
        // "set cwnd to ssthresh plus 3 * SMSS." (RFC 2581)
        conn->emit(cwndSignal, state->snd_cwnd);

        EV_DETAIL << " set cwnd=" << state->snd_cwnd << ", ssthresh=" << state->ssthresh << "\n";

        // Fast Retransmission: retransmit missing segment without waiting
        // for the REXMIT timer to expire
        // Do not restart REXMIT timer.
        // Note: Restart of REXMIT timer on retransmission is not part of RFC 2581, however optional in RFC 3517 if sent during recovery.
        // Resetting the REXMIT timer is discussed in RFC 2582/3782 (NewReno) and RFC 2988.

        // RFC 3517, page 7: "(4) Run SetPipe ()
        //
        // Set a "pipe" variable  to the number of outstanding octets
        // currently "in the pipe"; this is the data which has been sent by
        // the TCP sender but for which no cumulative or selective
        // acknowledgment has been received and the data has not been
        // determined to have been dropped in the network.  It is assumed
        // that the data is still traversing the network path."
        //conn->setPipe();
        // RFC 3517, page 7: "(5) In order to take advantage of potential additional available
        // cwnd, proceed to step (C) below."
        if (state->sack_enabled) {
            if (state->lossRecovery) {
                EV_INFO << "Retransmission sent during recovery, restarting REXMIT timer.\n";
                restartRexmitTimer();
            }
        }

        // try to transmit new segments (RFC 2581)
    }
    else if (state->dupacks > state->dupthresh) {
        //
        // Cubic: For each additional duplicate ACK received, increment cwnd by SMSS.
        // This artificially inflates the congestion window in order to reflect the
        // additional segment that has left the network
        //
        //state->snd_cwnd += state->snd_mss;
        EV_DETAIL << "Cubic on dupAcks > DUPTHRESH(=" << state->dupthresh << ": Fast Recovery: inflating cwnd by SMSS, new cwnd=" << state->snd_cwnd << "\n";

        //conn->emit(cwndSignal, state->snd_cwnd);

        // Note: Steps (A) - (C) of RFC 3517, page 7 ("Once a TCP is in the loss recovery phase the following procedure MUST be used for each arriving ACK")
        // should not be used here!

        // RFC 3517, pages 7 and 8: "5.1 Retransmission Timeouts
        // (...)
        // If there are segments missing from the receiver's buffer following
        // processing of the retransmitted segment, the corresponding ACK will
        // contain SACK information.  In this case, a TCP sender SHOULD use this
        // SACK information when determining what data should be sent in each
        // segment of the slow start.  The exact algorithm for this selection is
        // not specified in this document (specifically NextSeg () is
        // inappropriate during slow start after an RTO).  A relatively
        // straightforward approach to "filling in" the sequence space reported
        // as missing should be a reasonable approach."
    }

    if(state->snd_cwnd > 0){
        double paceFactor;
        if (state->snd_cwnd < state->ssthresh/2) {
            paceFactor = 2;
        }
        else{
            paceFactor = 1.2;
        }
       uint32_t maxWindow = std::max(state->snd_cwnd, dynamic_cast<TcpPacedConnection*>(conn)->getBytesInFlight());
       double pace = state->srtt.dbl()/((double) (maxWindow*paceFactor)/(double)state->snd_mss);
       // ORCA: Commented out pacing rate change here!
       //dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(pace);
    }

    sendData(false);
}


#endif