#include "omnetpp/ccomponent.h"
#include "omnetpp/simtime_t.h"
#include "transportlayer/tcp/TcpPacedConnection.h"
#include "transportlayer/tcp/flavours/TcpCubic.h"
#include <numeric>
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

        if(state->snd_cwnd >= state->ssthresh) {
            this->first_slowstart_complete=true;
        }
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
    this->rewardDelayForgiveness = this->conn->getTcpMain()->par("rewardDelayForgiveness");
    this->rewardLossMultiplier = this->conn->getTcpMain()->par("rewardLossMultiplier");
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
    scheduleNextStep(this->initialStepLength);
    // Schedule the first RL step
    // RLStep = new cMessage("RLSTEP");
    // conn->scheduleAt(simTime() + RLStepInterval, RLStep);
}

// OMNet Method? Called after component initialization is complete?
void Orca::established(bool active) {
    if (debug) cout << "\tOrca: established()" << endl;
    TcpCubic::established(active);
    dynamic_cast<TcpPacedConnection*>(conn)->subscribe(dynamic_cast<TcpPacedConnection*>(conn)->retransmissionRateSignal, (cListener*) this);
    if (active) {
        std::string s("Orca");
        setStringId(s);
        this->isActive = active;
    }
}







// Perform and observation and store the result into the provided vector (or append to it, if you're keeping history)
ObsType Orca::computeObservation(){
    if (debug) cout << "\tOrca: computeObservation()" << endl; 
    if (this->first_slowstart_complete == false) {
        if (debug) cout << "First slowstart not complete - skipping obs" << endl;
        return {0, 0, 0, 0, 0, 0, 0};
    }
    if (done) {
        cout << "Agent reported as done, skipping this obs" << endl;
        return {0, 0, 0, 0, 0, 0, 0};
    }
    dynamic_cast<TcpPacedConnection*>(conn)->computeRetransmissionRate(); // Updates this->retransmissionBytes via TcpPaced Connection
    double delta_snd_max = state->snd_max - this->last_snd_max;
    double delta_snd_una = state->snd_una - this->last_snd_una;
    this->orcaIntervalDuration = (simTime() - this->lastIntervalTime).dbl();

    // Throughput: How many bytes were DELIVERED this interval (basically goodput?)
    this->orcaThroughput = delta_snd_una / this->orcaIntervalDuration;
    this->orcaMaxThroughput = std::max(this->orcaMaxThroughput, this->orcaThroughput);

    // Lossrate: What percentage of bytes sent this interval were retransmissions
    this->orcaLossRate = 0.0;
    if (this->retransmissionRate > 0.0) {  // Avoid division by 0
        double transmissionRate = delta_snd_max/this->orcaIntervalDuration; // How many non-retransmits occurred this interval
        this->orcaLossRate = this->retransmissionRate / (this->retransmissionRate + transmissionRate);
    }
    // ACKed: How many bytes were ACKed this interval (basically raw goodput?)
    this->orcaACKTotal= delta_snd_una;
    //this->maxACKTotal = std::max(this->maxACKTotal, this->orcaACKTotal);

    // SRTT: Smoothed round trip time. Already tracked by TCP.
    this->orcaSRTT = state->srtt.dbl();

    // CWND: Size of the congestion window. Already tracked by TCP.
    this->orcaCwnd = (double) state->snd_cwnd;
    this->maxCwnd = std::max(this->maxCwnd, this->orcaCwnd);
    
    // Delay: Tracked in overridden method above. Only update the minimum if delay reports were received this interval.
    if (this->rttReportCount > 0) {
        this->orcaMinDelay = std::min(this->orcaMinDelay, this->orcaDelay);
    }

    // Delay Metric: The delay metric is treated as optimal if within the forgiveness window. Otherwise, have it slowly decrease as delay inflates.
    this->orcaDelayMetric = 1.0;
    if (this->orcaDelay > this->orcaMinDelay * this->rewardDelayForgiveness) {                                             
        this->orcaDelayMetric = this->orcaMinDelay * this->rewardDelayForgiveness / state->srtt;
    }
    
    if (this->orcaACKTotal == 0 || done) {
        if (debug) cout << "No packets ACKed. Skipping this observation." << endl;
        return {0, 0, 0, 0, 0, 0, 0};
    }

    // Should be:
    //      Throughput/max_bw
    //      Pace_rate/max_bw
    //      loss_rate/max_bw
    //      ACKs/cwnd
    //      interval_time (raw)
    //      min_rtt/srtt
    //      relaxed_min_rtt/srtt (1 if within delay margin)

    return {this->orcaThroughput / this->orcaMaxThroughput,     // Normalized throughput
            this->orcaPaceRate / this->orcaMaxThroughput,       // Normalized pacerate
            this->retransmissionRate / this->orcaMaxThroughput, // Normalized lossrate
            this->orcaACKTotal /  state->snd_cwnd,              // Normalized ACKs count (maybe use tcp_cwnd? ask aiden)     
            this->orcaIntervalDuration,                         // Monitor interval duration
            this->orcaMinDelay / this->orcaSRTT,                // Normalized SRTT (delay)
            this->orcaDelayMetric                               // Normalized SRTT (possibly forgiven, if within the forgiveness window)
        };

    // return {this->orcaThroughput / this->orcaMaxThroughput,
    //         this->orcaLossRate, // Loss rate, normalized as percentage of bits sent. Max to prevent division by 0.
    //         this->orcaDelay / this->orcaMinDelay,
    //         std::log(this->orcaACKTotal + 1), 
    //         this->orcaIntervalDuration, 
    //         this->orcaSRTT / this->orcaMinDelay, 
    //         std::log(this->orcaCwnd + 1),                // maybe should do    this->orcaCwnd / this->maxCwnd,
    //         std::log(this->orcaMaxThroughput + 1),       // log to scale values to reasonable range. +1e-6 to prevent log(0)
    //         this->orcaMinDelay
    //     };
}

RewardType Orca::computeReward(){
    if (debug) cout << "\tOrca: computeReward()" << endl;
    // Do not compute a reward if no ACKs were received. No ACKs means no throughput, no valid RTT measurement, etc.
    // Currently this just returns a 0 reward. TODO: Find a way to skip the RLStep altogether.
    // Note to self - maybe just don't return reward/obs, and instead schedule a new event? Something the upper layers won't see.
    if (this->rttReportCount == 0 || done || !this->first_slowstart_complete) {
        return RewardType(0.0);
    }
    // // Reward calculation: Reward the agent based on their proximity to the optimal throughput/delay ratio. (power)
    //     // Delay: If the measured delay is within some forgiveness window, then it does not negatively impact reward. Forgiveness window determined by rewardDelayForgiveness.
    //     // Loss: Loss directly subtracts from the rewards gained from thoughput. Strength of effect determined by rewardLossMultiplier.
    // double optimalPower = (this->orcaMaxThroughput/this->orcaMinDelay);         // Max possible reward based on observed max/min throughput/delay so far.
    // double currentPower;                                                        // Our actual measured reward for this interval
    // if (this->orcaDelay <= this->orcaMinDelay *this->rewardDelayForgiveness) {
    //     currentPower = (this->orcaThroughput - this->orcaLossRate*this->rewardLossMultiplier) / this->orcaMinDelay;   // Delay forgiven
    // } else {                                                                    
    //     currentPower = (this->orcaThroughput - this->orcaLossRate*this->rewardLossMultiplier) / this->orcaDelay;      // Delay NOT forgiven
    // }
    // double normalizedPower = currentPower / optimalPower; // How close this reward is to optimal. (0 is worst, 1 is optimal)
    // return RewardType(normalizedPower);

    return( (this->orcaThroughput-(this->rewardLossMultiplier*this->orcaLossRate))/this->orcaMaxThroughput*this->orcaDelayMetric);
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void Orca::decisionMade(ActionType action) {
    scheduleNextStep(state->srtt.dbl()); // Schedule the next RLStep
    if (debug) cout << "\tOrca: decisionMade()" << endl;
    RLStepsTaken++;
    if (debug) cout << "\t\tRLSteps taken: " << RLStepsTaken << endl;
    if (RLStepsTaken >= this->maxRLSteps) {
            if (debug) cout << "\t\tWE ARE DONE! " << RLStepsTaken << " STEPS TAKEN!" << endl;
            done = true; // Don't set done yourself. Unsure of the correct way to handle this, but this isn't it.
    }
    if (this->orcaACKTotal == 0) {
        if (debug) cout << "No packets ACK'd this interval. Skipping action, cwnd staying at " << state->snd_cwnd << endl;
        return;
    } 

    // Avoid taking actions until initial slowstart is complete
    if (this->first_slowstart_complete == false) {
        if (debug) cout << "Currently in slow start. Orca will not apply any action.";
        return;
    }
        double fakeAction = action;
        uint32_t newCwnd = ceil(std::pow(2.0, fakeAction) * (double) state->snd_cwnd);
        newCwnd =  max(state->snd_mss, newCwnd);
        // dont let cwnd inflate to ridiculous values. Learning will take care of this eventually, but large values eventually kill simulations.
        if (newCwnd < 1000000) {
            if (debug) cout << "\t\tChanging cwnd from " << state->snd_cwnd << " to " << newCwnd << "(" << (double)newCwnd/(double)state->snd_cwnd << "x)" << endl;
            state->snd_cwnd = newCwnd;
        }
        

        double newIntersendingTime = state->srtt.dbl() / (double) state->snd_cwnd;  // Pace rate expressed as seconds between packets (cwnd/srtt per second)
        
        // cout << "srtt: " << state->srtt.dbl() << endl;
        // cout << "interSendTime: " << newIntersendingTime << endl;
        orcaPaceRate = (double) state->snd_cwnd / state->srtt.dbl();  // Bytes/s
        dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(1/orcaPaceRate); // Time between bytes

        // Change the stepSize to be 1 RTT (based on srtt)
        // cObject* newStepSizeObj = new cSimTime(state->srtt.dbl());
        // cout << "\t\tChanging step size to " << newStepSizeObj << endl;
        
        // owner->emit(this->modifyStepSizeSig, stringId.c_str(), newStepSizeObj); 
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
    this->last_snd_max = state->snd_max;
    this->last_snd_una = state->snd_una;
    this->lastIntervalTime = simTime();
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