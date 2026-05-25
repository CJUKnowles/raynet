#include "TcpPacedNoCC.h"

namespace inet {
namespace tcp {

Register_Class(TcpPacedNoCC);

void TcpPacedNoCC::processRexmitTimer(TcpEventCode &event) {
    TcpPacedFamily::processRexmitTimer(event);

    std::cerr << "RTO at " << simTime() << std::endl;
    std::cerr << "cwnd=: " << state->snd_cwnd / state->snd_mss << ", in-flight="
            << (state->snd_max - state->snd_una) / state->snd_mss << std::endl;


    reset();
    recalculateSlowStartThreshold();
    // state->snd_cwnd = state->snd_mss; // TcpPacedNoCC: do NOT alter snd_cwnd

//    if(state->snd_cwnd > 0){
//        if(state->snd_cwnd < state->ssthresh/2){
//            dynamic_cast<PacedTcpConnection*>(conn)->changeIntersendingTime(state->srtt.dbl()/(((double) state->snd_cwnd/(double)state->snd_mss)* 2));
//        }
//        else{
//            dynamic_cast<PacedTcpConnection*>(conn)->changeIntersendingTime(state->srtt.dbl()/(((double) state->snd_cwnd/(double)state->snd_mss)* 2));
//        }
//    }

    conn->emit(cwndSignal, state->snd_cwnd);

    state->afterRto = true;
    dynamic_cast<TcpPacedConnection*>(conn)->cancelPaceTimer();
    sendData(false);

    conn->emit(ssthreshSignal, state->ssthresh);
    conn->emit(cwndSegSignal, state->snd_cwnd / state->snd_mss);
}

void TcpPacedNoCC::receivedDataAck(uint32_t firstSeqAcked) {

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
            // state->snd_cwnd = state->ssthresh; // TcpPacedNoCC: do NOT alter snd_cwnd
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
        // state->snd_cwnd += state->snd_mss; // TcpPacedNoCC: do NOT alter snd_cwnd
        conn->emit(cwndSignal, state->snd_cwnd);
        conn->emit(ssthreshSignal, state->ssthresh);

        EV_INFO << "cwnd=" << state->snd_cwnd << "\n";
    }
    else {

        updateCubicCwnd(1);

        if (state->cwnd_cnt >= state->cnt) {
            // state->snd_cwnd += state->snd_mss; // TcpPacedNoCC: do NOT alter snd_cwnd
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
        // dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(state->srtt.dbl()/(((double) maxWindow/(double)state->snd_mss)* paceFactor)); // TcpPacedNoCC: Do not alter pacing rate automatically
    }

    sendData(false);

    conn->emit(cwndSegSignal, state->snd_cwnd / state->snd_mss);
}

void TcpPacedNoCC::receivedDuplicateAck()
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
                // state->snd_cwnd = state->ssthresh + (3*state->snd_mss); // TcpPacedNoCC: do NOT alter snd_cwnd
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
       // dynamic_cast<TcpPacedConnection*>(conn)->changeIntersendingTime(pace); // TcpPacedNoCC: Do not alter pacing rate automatically
    }

    sendData(false);
}

} // inet
} // tcp