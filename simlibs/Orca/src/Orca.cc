#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/flavours/TcpCubic.h"
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
    // if (RLStep) {
    //     delete cancelEvent(RLStep);
    // }
    getSimulation()->getSystemModule()->unsubscribe(stringId.c_str(), (cListener*) this);
    getSimulation()->getSystemModule()->unsubscribe("actionResponse", (cListener*) this);
}

// Override receivedDataAck from TcpCubic to remove default pacing behaviour (pacing rate should only change once per monitor interval!')
void Orca::receivedDataAck(uint32_t firstSeqAcked) {

    TcpTahoeRenoFamily::receivedDataAck(firstSeqAcked);
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
    // > pacing functionality removed from right here <
    sendData(false);

    conn->emit(cwndSegSignal, state->snd_cwnd / state->snd_mss);
}

// Override receivedDuplicateAck from TcpCubic to remove default pacing functionality (pacing rate should only change once per RLStep)
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
    // > Pacing removed from right here <
    sendData(false);
}

// Called upon a valid ACK received (?); Grab the RTT measured and use it to update the current interval's average (may be faster to store all values and average at the end of the interval)
void Orca::rttMeasurementComplete(simtime_t tSent, simtime_t tAcked) {
    TcpCubic::rttMeasurementComplete(tSent, tAcked);
    double packetRTT = (tAcked-tSent).dbl();
    this->orcaDelay = (this->orcaDelay * (double) rttReportCount + packetRTT) / (rttReportCount + 1);
    this->rttReportCount += 1;
}




// // RayNet: Called to initalize the agent
void Orca::initialize() {
    if (debug) cout << "\tOrca initialize()" << endl;
    int _stateSize = this->conn->getTcpMain()->par("stateSize");;
    int _maxObsCount = this->conn->getTcpMain()->par("maxObsCount");
    this->delayWeight = this->conn->getTcpMain()->par("delayWeight");
    this->maxRLSteps = this->conn->getTcpMain()->par("maxRLSteps");
    debug = this->conn->getTcpMain()->par("printDebugMessages");

    // provide the RLInterface with a cComponent API (to use signaling functionality)
    setOwner((cComponent*) conn->getTcpMain());
    
    // Initalize parent classes
    // RLInterface::initialize(_stateSize, _maxObsCount); // Deprecated initialization function. Delete this later.
    RLInterface::initialise();
    TcpCubic::initialize();

    // Set the RL ID of this component (for use by the training script). Ensure this is unique for multi-agent environments (perhaps use the IP of the host?)
    std::string s("Orca");
    setStringId(s);
    
    // Register this agent with RayNet
    cObject* simtime = new cSimTime(this->conn->getTcpMain()->par("monitorIntervalDuration"));
    owner->emit(this->registerSig, stringId.c_str(), simtime); 

    // Schedule the first RL step
    // RLStep = new cMessage("RLSTEP");
    // conn->scheduleAt(simTime() + RLStepInterval, RLStep);
}

// OMNet Method? Called after component initialization is complete?
void Orca::established(bool active) {
    if (debug) cout << "\tOrca: established()" << endl;
    TcpCubic::established(active);

    if (active) {
        std::string s("Orca");
        setStringId(s);
        this->isActive = active;
    }
}







// Perform and observation and store the result into the provided vector (or append to it, if you're keeping history)
ObsType Orca::computeObservation(){
    if (debug) cout << "\tOrca: computeObservation()" << endl; 
    
    this->orcaIntervalDuration = (simTime() - this->lastIntervalTime).dbl();
    this->orcaThroughput = (state->snd_max - this->lastIntervalSentBytes) / this->orcaIntervalDuration;
    this->orcaLossRate=0.0;             // Track total sent and total lost. Perform final division here.
    this->orcaACKTotal= state->snd_una - this->lastIntervalSndUna;  // Check how many ACK's occured this interval (see how many packets snd_una has increased by)
    this->orcaSRTT = state->srtt.dbl();
    this->orcaCwnd = (double) state->snd_cwnd;
    this->orcaMaxThroughput = std::max(this->orcaMaxThroughput, this->orcaThroughput);
    if (this->rttReportCount > 0) {
        // Only update the minDelay if an ACK has been receieved this interval.
        // This is done to prevent division by zero.
        // At some point I need to implement skipping if this happens.
        this->orcaMinDelay = std::min(this->orcaMinDelay, this->orcaDelay);
    }
    this->maxCwnd = std::max(this->maxCwnd, this->orcaCwnd);
    this->maxACKTotal = std::max(this->maxACKTotal, this->orcaACKTotal);

    // Should I update these in resetStepVariables? How much later is that called?
    this->lastIntervalSndUna = state->snd_una;
    this->lastIntervalSentBytes = state->snd_max;
    this->lastIntervalTime = simTime();
    return {this->orcaThroughput / this->orcaMaxThroughput,
            this->orcaLossRate, // not implemented yet
            this->orcaDelay / this->orcaMinDelay,
            this->orcaACKTotal / this->maxACKTotal, 
            this->orcaIntervalDuration, 
            this->orcaSRTT / this->orcaMinDelay, 
            this->orcaCwnd / this->maxCwnd,
            this->orcaMaxThroughput, 
            this->orcaMinDelay
        };
}

RewardType Orca::computeReward(){
    if (debug) cout << "\tOrca: computeReward()" << endl;
    // Do not compute a reward if no ACKs were received. No ACKs means no throughput, no valid RTT measurement, etc.
    // Currently this just returns a 0 reward. TODO: Find a way to skip the RLStep altogether.
    if (this->rttReportCount == 0) {
        return RewardType(0.0);
    }
    // Reward calculation: Reward the agent based on their proximity to the optimal throughput/delay ratio. (power)
    // If the current delay is reasonably close to optimal, it will be treated as optimal.
    double optimalPower = (this->orcaMaxThroughput/this->orcaMinDelay);     // Max possible reward based on observed max/min throughput/delay so far.
    double currentPower;                                                    // Our actual measured reward for this interval
    if (this->orcaDelay <= this->delayWeight * this->orcaMinDelay) {
        currentPower = this->orcaThroughput / this->orcaMinDelay;
    } else {
        currentPower = this->orcaThroughput / this->orcaDelay;
    }
    double normalizedPower = currentPower / optimalPower;                   // How close this reward is to optimal. (0 is worst, 1 is optimal)
    return RewardType(normalizedPower);
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void Orca::decisionMade(ActionType action) {
    if (debug) cout << "\tOrca: decisionMade()" << endl;

    if (!isnan(action) && isActive) {
        if (debug) cout << "\t\tAction received: " << action << endl;

        if (isReset) {
            if (debug) cout << "\t\tOrca currently resetting, will not take action" << endl;
        } else {
            // Change the current cwnd based on the action. Do not let it drop below the maximum segment size.
            if (this->orcaACKTotal == 0) {
                if (debug) cout << "No packets ACK'd this interval. Skipping action, cwnd staying at " << state->snd_cwnd << endl;
            } else {
                double fakeAction = action;
                uint32_t newCwnd = ceil(std::pow(2.0, fakeAction) * (double) state->snd_cwnd);
                newCwnd =  max(state->snd_mss, newCwnd);
                state->snd_cwnd = newCwnd;
                double newIntersendingTime = state->srtt.dbl() / (double) state->snd_cwnd;  // Pace rate expressed as seconds between packets (cwnd/srtt per second)
                
                cout << "srtt: " << state->srtt.dbl() << endl;
                cout << "interSendTime: " << newIntersendingTime << endl;
                dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(newIntersendingTime);

                // Change the stepSize to be 1 RTT (based on srtt)
                // cObject* newStepSizeObj = new cSimTime(state->srtt.dbl());
                // cout << "\t\tChanging step size to " << newStepSizeObj << endl;
                
                // owner->emit(this->modifyStepSizeSig, stringId.c_str(), newStepSizeObj); 

                this->modifyStepSize(state->srtt.dbl());
            }
        }

        RLStepsTaken++;
        if (debug) cout << "\t\tRLSteps taken: " << RLStepsTaken << endl;
        if (RLStepsTaken >= this->maxRLSteps) {
            if (debug) cout << "\t\tWE ARE DONE! " << RLStepsTaken << " STEPS TAKEN!" << endl;
            done = true; // Don't set done yourself. Unsure of the correct way to handle this, but this isn't it.
        }
    }
    else {
        EV_ERROR << action << " value in decisionMade() function" << std::endl;
    }
}


void Orca::resetStepVariables()
{
    if (debug) cout << "\t\tOrca: resetStepVariables()" << endl;
    this->orcaThroughput=0.0;    // The average delivery rate (throughput) over the last interval
    this->orcaLossRate=0.0;      // The average loss rate of packets over the last interval
    this->orcaDelay=0.0;         // The average delay of packets over the last interval
    this->orcaACKTotal=0.0;      // The number of valid acknowledgements over the last interval
    this->orcaIntervalDuration=0.0;  // The simtime elapsed over the last interval

    this->rttReportCount=0; // The number of RTT values we have measured over the last interval
}

// Returns true if the agent is reporting this episode as complete. (Pretty sure this is never called. Just set done to true directly during an RLStep.)
bool Orca::getDone() {
    if (debug) cout << "Orca getDone(): If you're seeing this, getDone() probably isn't deprecated.";
    bool done = RLStepsTaken > 1000;
    if (debug) cout << "\tOrca: " << RLStepsTaken << " steps completed. Returning " << done << endl;
    return done;
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


#endif