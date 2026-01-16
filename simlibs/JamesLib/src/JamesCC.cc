#include "JamesCC.h"


using namespace inet::tcp;
using namespace inet;
using namespace learning;

Register_Class(JamesCC); // Lets omnet see and use this class

JamesCC::JamesCC():
    TcpNewReno(), RLInterface() {
    if (debug) cout << "\tJamesCC: Constructor called!";
}
JamesCC::~JamesCC() {
    if (debug) cout << "\tJamesCC: Destructor method called. Goodbye.";
    if (RLStep) {
        delete cancelEvent(RLStep);
    }
}

// CWND is the current congestion window size, dictating how many packets are allowed in-flight
// ssthresh is the CWND value at which the window's growth switches from exponential (slow start) to linear (congestion avoidance)
    // Think of ssthresh as "remembering" the safe capacity of the network. It tells the connection when to slow down CWND growth as it is approaching previous levels of loss.


// ----- Slow Start (exponential) -----
// CWND increases exponentially with each ACK 
// This continues until CWND exceeds ssthresh (or loss is detected)
// At this point, slow start transitions to congesiton avoidance (growth slows from exponential to linear)

// ----- Congestion Avoidance (linear) -----
// CWND increases linearly with each ACK
// This continues until loss is detected 
// 

// ----- Timeout Event (reset) -----
// No ACKs for a given timeout period triggers this event
// CWND is reset to 1, and 

// (Generally) cuts ssthresh in half, making the slower, linear increase begin sooner
void JamesCC::recalculateSlowStartThreshold()
{
    //if (debug) cout << "\tJamesCC recalculateSlowStartThreshold()" << endl;
    uint32_t flight_size = std::min(state->snd_cwnd, state->snd_wnd); 
//    uint32_t flight_size = state->snd_max - state->snd_una;
    state->ssthresh = std::max(flight_size / 2, 2 * state->snd_mss);

    conn->emit(ssthreshSignal, state->ssthresh);
}

// Timeout - Reset cwnd, reduce ssthresh, enter slow start
void JamesCC::processRexmitTimer(TcpEventCode& event)
{
    TcpTahoeRenoFamily::processRexmitTimer(event);
    if (event == TCP_E_ABORT)
        return;

    // Record highest seq sent. Exit loss recovery mode if currently in it (triple duplicate ACK)
    state->recover = (state->snd_max - 1);              // highest seq# transmitted
    EV_INFO << "recover=" << state->recover << "\n";
    state->lossRecovery = false;
    state->firstPartialACK = false;
    EV_INFO << "Loss Recovery terminated.\n";

    // Enter slow start phase (lower ssthresh, reset cwnd, send first packet)
    recalculateSlowStartThreshold();    // Multiplicitive decrease (reduce ssthresh)
    state->snd_cwnd = state->snd_mss;   // Reset cwnd to 1
    conn->emit(cwndSignal, state->snd_cwnd);
    EV_INFO << "Begin Slow Start: resetting cwnd to " << state->snd_cwnd
            << ", ssthresh=" << state->ssthresh << "\n";
    state->afterRto = true;
    conn->retransmitOneSegment(true);
}

// ACK received - increase cwnd and ssthresh
void JamesCC::receivedDataAck(uint32_t firstSeqAcked)
{
    TcpTahoeRenoFamily::receivedDataAck(firstSeqAcked);

    // RFC 3782, page 5:
    // "5) When an ACK arrives that acknowledges new data, this ACK could be
    // the acknowledgment elicited by the retransmission from step 2, or
    // elicited by a later retransmission.
    //
    // Full acknowledgements:
    // If this ACK acknowledges all of the data up to and including
    // "recover", then the ACK acknowledges all the intermediate
    // segments sent between the original transmission of the lost
    // segment and the receipt of the third duplicate ACK.  Set cwnd to
    // either (1) min (ssthresh, FlightSize + SMSS) or (2) ssthresh,
    // where ssthresh is the value set in step 1; this is termed
    // "deflating" the window.  (We note that "FlightSize" in step 1
    // referred to the amount of data outstanding in step 1, when Fast
    // Recovery was entered, while "FlightSize" in step 5 refers to the
    // amount of data outstanding in step 5, when Fast Recovery is
    // exited.)  If the second option is selected, the implementation is
    // encouraged to take measures to avoid a possible burst of data, in
    // case the amount of data outstanding in the network is much less
    // than the new congestion window allows.  A simple mechanism is to
    // limit the number of data packets that can be sent in response to
    // a single acknowledgement; this is known as "maxburst_" in the NS
    // simulator.  Exit the Fast Recovery procedure."

    // In loss recovery: Check if the ACK'd segment is full (ACK's all missing segments) or partial (there are stil packets missing)
    if (state->lossRecovery) {
        // Full ACK - Missing packets have arrived, fully deflate cwnd to normal levels and resume normal operation
        if (seqGE(state->snd_una - 1, state->recover)) {
            // Exit Fast Recovery: deflating cwnd
            //
            // option (1): set cwnd to min (ssthresh, FlightSize + SMSS)
            uint32_t flight_size = state->snd_max - state->snd_una;
            state->snd_cwnd = std::min(state->ssthresh, flight_size + state->snd_mss);
            EV_INFO << "Fast Recovery - Full ACK received: Exit Fast Recovery, setting cwnd to " << state->snd_cwnd << "\n";
            // option (2): set cwnd to ssthresh
//            state->snd_cwnd = state->ssthresh;
//            tcpEV << "Fast Recovery - Full ACK received: Exit Fast Recovery, setting cwnd to ssthresh=" << state->ssthresh << "\n";
            // TODO - If the second option (2) is selected, take measures to avoid a possible burst of data (maxburst)!
            conn->emit(cwndSignal, state->snd_cwnd);

            state->lossRecovery = false;
            state->firstPartialACK = false;
            EV_INFO << "Loss Recovery terminated.\n";
        }
        // Partial ACK - More packets are missing. Retransmit the next missing segment, and partially deflate cwnd to account for the ACK'd segment
        else {
            // Retransmit first next missing segment
            EV_INFO << "Fast Recovery - Partial ACK received: retransmitting the first unacknowledged segment\n";         
            conn->retransmitOneSegment(false);        // retransmit first unacknowledged segment

            // Deflate cwnd proportial to ACK'd data
            state->snd_cwnd -= state->snd_una - firstSeqAcked;      // deflate cwnd by amount of new data acknowledged by cumulative acknowledgement field
            conn->emit(cwndSignal, state->snd_cwnd);
            EV_INFO << "Fast Recovery: deflating cwnd by amount of new data acknowledged, new cwnd=" << state->snd_cwnd << "\n";

            // Re-inflate cwnd by 1 to make room for another retransmission
            if (state->snd_una - firstSeqAcked >= state->snd_mss) {
                state->snd_cwnd += state->snd_mss;
                conn->emit(cwndSignal, state->snd_cwnd);
                EV_DETAIL << "Fast Recovery: inflating cwnd by SMSS, new cwnd=" << state->snd_cwnd << "\n";
            }

            // try to send a new segment if permitted by the new value of cwnd
            sendData(false);

            // reset REXMIT timer for the first partial ACK that arrives during Fast Recovery
            if (state->lossRecovery) {
                if (!state->firstPartialACK) {
                    state->firstPartialACK = true;
                    EV_DETAIL << "First partial ACK arrived during recovery, restarting REXMIT timer.\n";
                    restartRexmitTimer();
                }
            }
        }
    }
    else {
        //
        // Perform slow start and congestion avoidance.
        //
        if (state->snd_cwnd < state->ssthresh) {
            EV_DETAIL << "cwnd <= ssthresh: Slow Start: increasing cwnd by SMSS bytes to ";

            // perform Slow Start. RFC 2581: "During slow start, a TCP increments cwnd
            // by at most SMSS bytes for each ACK received that acknowledges new data."
            state->snd_cwnd += state->snd_mss; // James TODO: vary the amount snd_cwnd increases based on the policy

            // Note: we could increase cwnd based on the number of bytes being
            // acknowledged by each arriving ACK, rather than by the number of ACKs
            // that arrive. This is called "Appropriate Byte Counting" (ABC) and is
            // described in RFC 3465. This RFC is experimental and probably not
            // implemented in real-life TCPs, hence it's commented out. Also, the ABC
            // RFC would require other modifications as well in addition to the
            // two lines below.
            //
//            int bytesAcked = state->snd_una - firstSeqAcked;
//            state->snd_cwnd += bytesAcked * state->snd_mss;

            conn->emit(cwndSignal, state->snd_cwnd);

            EV_DETAIL << "cwnd=" << state->snd_cwnd << "\n";
        }
        else {
            // perform Congestion Avoidance (RFC 2581)
            uint32_t incr = state->snd_mss * state->snd_mss / state->snd_cwnd;

            if (incr == 0)
                incr = 1;

            state->snd_cwnd += incr;

            conn->emit(cwndSignal, state->snd_cwnd);

            //
            // Note: some implementations use extra additive constant mss / 8 here
            // which is known to be incorrect (RFC 2581 p5)
            //
            // Note 2: RFC 3465 (experimental) "Appropriate Byte Counting" (ABC)
            // would require maintaining a bytes_acked variable here which we don't do
            //

            EV_DETAIL << "cwnd > ssthresh: Congestion Avoidance: increasing cwnd linearly, to " << state->snd_cwnd << "\n";
        }

        // RFC 3782, page 13:
        // "When not in Fast Recovery, the value of the state variable "recover"
        // should be pulled along with the value of the state variable for
        // acknowledgments (typically, "snd_una") so that, when large amounts of
        // data have been sent and acked, the sequence space does not wrap and
        // falsely indicate that Fast Recovery should not be entered (Section 3,
        // step 1, last paragraph)."
        state->recover = (state->snd_una - 2);
    }

    sendData(false);
    // Data has been sent. Peform an RL step.
    
}

// Duplicate ACK received - attempt a fast restransmit
void JamesCC::receivedDuplicateAck()
{
    TcpTahoeRenoFamily::receivedDuplicateAck();

    if (state->dupacks == state->dupthresh) {
        if (!state->lossRecovery) {
            // RFC 3782, page 4:
            // "1) Three duplicate ACKs:
            // When the third duplicate ACK is received and the sender is not
            // already in the Fast Recovery procedure, check to see if the
            // Cumulative Acknowledgement field covers more than "recover".  If
            // so, go to Step 1A.  Otherwise, go to Step 1B."
            //
            // RFC 3782, page 6:
            // "Step 1 specifies a check that the Cumulative Acknowledgement field
            // covers more than "recover".  Because the acknowledgement field
            // contains the sequence number that the sender next expects to receive,
            // the acknowledgement "ack_number" covers more than "recover" when:
            //      ack_number - 1 > recover;"
            if (state->snd_una - 1 > state->recover) {
                EV_INFO << "NewReno on dupAcks == DUPTHRESH(=" << state->dupthresh << ": perform Fast Retransmit, and enter Fast Recovery:";

                // RFC 3782, page 4:
                // "1A) Invoking Fast Retransmit:
                // If so, then set ssthresh to no more than the value given in
                // equation 1 below.  (This is equation 3 from [RFC2581]).
                //      ssthresh = max (FlightSize / 2, 2*SMSS)           (1)
                // In addition, record the highest sequence number transmitted in
                // the variable "recover", and go to Step 2."
                recalculateSlowStartThreshold();
                state->recover = (state->snd_max - 1);
                state->firstPartialACK = false;
                state->lossRecovery = true;
                EV_INFO << " set recover=" << state->recover;

                // RFC 3782, page 4:
                // "2) Entering Fast Retransmit:
                // Retransmit the lost segment and set cwnd to ssthresh plus 3 * SMSS.
                // This artificially "inflates" the congestion window by the number
                // of segments (three) that have left the network and the receiver
                // has buffered."
                state->snd_cwnd = state->ssthresh + 3 * state->snd_mss;

                conn->emit(cwndSignal, state->snd_cwnd);

                EV_DETAIL << " , cwnd=" << state->snd_cwnd << ", ssthresh=" << state->ssthresh << "\n";
                conn->retransmitOneSegment(false);

                // RFC 3782, page 5:
                // "4) Fast Recovery, continued:
                // Transmit a segment, if allowed by the new value of cwnd and the
                // receiver's advertised window."
                sendData(false);
            }
            else {
                EV_INFO << "NewReno on dupAcks == DUPTHRESH(=" << state->dupthresh << ": not invoking Fast Retransmit and Fast Recovery\n";

                // RFC 3782, page 4:
                // "1B) Not invoking Fast Retransmit:
                // Do not enter the Fast Retransmit and Fast Recovery procedure.  In
                // particular, do not change ssthresh, do not go to Step 2 to
                // retransmit the "lost" segment, and do not execute Step 3 upon
                // subsequent duplicate ACKs."
            }
        }
        EV_INFO << "NewReno on dupAcks == DUPTHRESH(=" << state->dupthresh << ": TCP is already in Fast Recovery procedure\n";
    }
    else if (state->dupacks > state->dupthresh) {
        if (state->lossRecovery) {
            // RFC 3782, page 4:
            // "3) Fast Recovery:
            // For each additional duplicate ACK received while in Fast
            // Recovery, increment cwnd by SMSS.  This artificially inflates the
            // congestion window in order to reflect the additional segment that
            // has left the network."
            state->snd_cwnd += state->snd_mss;

            conn->emit(cwndSignal, state->snd_cwnd);

            EV_DETAIL << "NewReno on dupAcks > DUPTHRESH(=" << state->dupthresh << ": Fast Recovery: inflating cwnd by SMSS, new cwnd=" << state->snd_cwnd << "\n";

            // RFC 3782, page 5:
            // "4) Fast Recovery, continued:
            // Transmit a segment, if allowed by the new value of cwnd and the
            // receiver's advertised window."
            sendData(false);
        }
    }

}


// // RayNet: Called to initalize the agent
void JamesCC::initialize() {
    if (debug) cout << "\tJamesCC initialize()" << endl;
    int _stateSize = this->conn->getTcpMain()->par("stateSize");;
    int _maxObsCount = this->conn->getTcpMain()->par("maxObsCount");
    debug = this->conn->getTcpMain()->par("printDebugMessages");
    
    // provide the RLInterface with a cComponent API (to use signaling functionality)
    setOwner((cComponent*) conn->getTcpMain());

    // Initalize parent classes
    // RLInterface::initialize(_stateSize, _maxObsCount); // Deprecated initialization function. Delete this later.
    RLInterface::initialise();
    TcpNewReno::initialize();

    // Set the RL ID of this component (for use by the training script). Ensure this is unique for multi-agent environments (perhaps use the IP of the host?)
    std::string s("JamesCC");
    setStringId(s);
    
    // Register this agent with RayNet
    cObject* simtime = new cSimTime(1);
    owner->emit(this->registerSig, stringId.c_str(), simtime); 

    // Schedule the first RL step
    // RLStep = new cMessage("RLSTEP");
    // conn->scheduleAt(simTime() + RLStepInterval, RLStep);
}

// OMNeT method that catches timers set by scheduleAt() and similar. Necessary for self-scheduling events.
// void JamesCC::processTimer(cMessage *timer, TcpEventCode &event) {
//     if (timer == RLStep) {
//         if (debug) cout << "\tJamesCC: Performing an RLStep!" << endl;
//         owner->emit(senderToStepper, this, new cString(stringId)); // Request the action! Maybe pass self?

//         // Schedule another RL step and increment the RLStep counter
//         conn->scheduleAt(simTime() + RLStepInterval, RLStep);
//         RLStepsTaken++;
//         if (RLStepsTaken > 100) {
//             if (debug) cout << "\t\tWE ARE DONE! " << RLStepsTaken << " STEPS TAKEN!" << endl;
//             done = true;
//         }
//     } else {
//         TcpNewReno::processTimer(timer, event);
//     }
// }

// // Compute an observation and populate the input vector with that data
// void JamesCC::getObservationVec(std::vector<double> &vec) {
//     //placeholder: Just throw in some random values. We are modifiyng an object passed by reference - no need to return anything
//     vec.push_back(1.0);
//     vec.push_back(2.0);
//     vec.push_back(3.0);
//     vec.push_back(4.0);
// }


// OMNet Method? Called after component initialization is complete?
void JamesCC::established(bool active) {
    if (debug) cout << "\tJamesCC: established()" << endl;
    TcpNewReno::established(active);

    if (active) {
        std::string s("JamesCC");
        setStringId(s);
        //setStringId(conn->getLocalAddress().str());
        this->isActive = active;
        conn->emit(cwndSignal, state->snd_cwnd);
        conn->emit(dupAcksSignal, dupAcks);
    }
}

// RayNet method: Called after simulation completion? Unsure how this differs from reset()
void JamesCC::cleanup()
{
    if (debug) cout << "\tJamesCC: cleanUp()" << endl;
}

// RayNet method: Make a decision based on the policy (alter snd_cwnd)
void JamesCC::decisionMade(ActionType action) {
    if (debug) cout << "\tJamesCC: decisionMade()" << endl;
    if (!isnan(action)) {

        if (debug) cout << "\t\tAction received: " << action << endl;
        // TODO: perform some action, like setting the congestion window
        // conn->emit(actionSignal, action);
        // conn->emit(cwndSignal, state->snd_cwnd);
        if (isReset) {
            cout << "\t\tJamesCC currently resetting, will not take action" << endl;
        } else {
            if (debug) cout << "\t\tJamesCC not resetting! Action being taken." << endl;
            //state->snd_cwnd = static_cast<uint32_t>(max((double) state->snd_mss, ceil(action * maxLearnWindow * (double) state->snd_mss)));
        }

        RLStepsTaken++;
        cout << "\t\tRLSteps taken: " << RLStepsTaken << endl;
        if (RLStepsTaken >= 20) {
            if (debug) cout << "\t\tWE ARE DONE! " << RLStepsTaken << " STEPS TAKEN!" << endl;
            //done = true; // Don't set done yourself. Unsure of the correct way to handle this, but this isn't it.
        }
    }
    else {
        EV_ERROR << action << " value in decisionMade() function" << std::endl;
    }
}


ObsType JamesCC::getRLState(){
    if (debug) cout << "\tJamesCC: getRLState()" << endl;
    return {0.0, 0.0, 0.0, 0.0};
    //return state;
}

RewardType JamesCC::getReward(){
    if (debug) cout << "\tJamesCC: getReward()" << endl;
    RewardType reward;
    reward = 1.0;
    return reward;
}
bool JamesCC::getDone(){
    cout << "\tJamesCC: getDone()" << endl;
    bool done = RLStepsTaken > 100;
    if (debug) cout << "\tJamesCC: " << RLStepsTaken << " steps completed. Returning " << done << endl;
    return done;
}
void JamesCC::resetStepVariables()
{
    if (debug) cout << "\tJamesCC: resetStepVariables()" << endl;
}

// Perform and observation and store the result into the provided vector (or append to it, if you're keeping history)
ObsType JamesCC::computeObservation(){
    if (debug) cout << "\tJamesCC: computeObservation()" << endl; 
    return getRLState();
}
RewardType JamesCC::computeReward(){
    if (debug) cout << "\tJamesCC: computeReward()" << endl;
    return getReward();
}

















// // // RayNet: Called to initalize the agent
// void JamesCC::initialize() {
//     if (debug) cout << "\tJamesCC initialize()" << endl;
//     int _stateSize = this->conn->getTcpMain()->par("stateSize");;
//     int _maxObsCount = this->conn->getTcpMain()->par("maxObsCount");

//     // Monitor intervals - allows our class to be monitored alongside other simulation components
//     miHandler = MonitorIntervalsHandler();
//     miHandler.currentMi = nullptr;
//     monitorInterval = new cMessage("MONITORINTERVAL");
    
//     // provide the RLInterface with a cComponent API (to use signaling functionality)
//     setOwner((cComponent*) conn->getTcpMain());

//     RLInterface::initialize(_stateSize, _maxObsCount);
//     TcpNewReno::initialize();

//     // dupAcks = 0;
//     // lastMIReward = -1;
//     // lastMIavgRTT = -1;
//     lastMiAction = 0;

//     throughputSignal = conn->registerSignal("throughput");
//     actionSignal = conn->registerSignal("action");
//     dupAcksSignal = conn->registerSignal("dupAcks");
//     rttGradientSignal = conn->registerSignal("rttGradient");
//     tickSignal = conn->registerSignal("tick");
//     miQueueSizeSignal = conn->registerSignal("MIQueueSize");

//     std::string s("JamesCC");
//     setStringId(s);

//     initMsg = new cMessage("CARTPOLE-INIT"); 
//     scheduleAt(simTime() + 1, initMsg);
// }

// void JamesCC::established(bool active) {
//     if (debug) cout << "\tJamesCC established()" << endl;
//     TcpNewReno::established(active);

//     if (active) {
//         std::string s("JamesCC");
//         setStringId(s);
//         //setStringId(conn->getLocalAddress().str());
//         this->isActive = active;
//         conn->emit(cwndSignal, state->snd_cwnd);
//         conn->emit(dupAcksSignal, dupAcks);
//     }
// }

// // void JamesCC::step(ActionType action)
// // {
// // }

// void JamesCC::cleanup()
// {
//     if (debug) cout << "\tJamesCC cleanUp()" << endl;
// }

// // Make a decision based on the policy (alter snd_cwnd)
// void JamesCC::decisionMade(ActionType action) {
//     if (debug) cout << "\tJamesCC decisionMade()" << endl;
//     if (!isnan(action)) {
//         //state->snd_cwnd = static_cast<uint32_t>(max((double) state->snd_mss, ceil(action * maxLearnWindow * (double) state->snd_mss)));
//         conn->emit(actionSignal, action);
//         conn->emit(cwndSignal, state->snd_cwnd);
//     }
//     else {
//         EV_ERROR << action << " value in decisionMade() function" << std::endl;
//     }
// }

// ObsType JamesCC::getRLState(){
//     if (debug) cout << "\tJamesCC getRLState()" << endl;
//     return {0.0, 0.0, 0.0, 0.0};
//     //return state;
// }

// RewardType JamesCC::getReward(){
//     if (debug) cout << "\tJamesCC getReward()" << endl;
//     RewardType reward;
//     reward = 1.0;
//     return reward;
// }
// bool JamesCC::getDone(){
//     if (debug) cout << "\tJamesCC getDone()" << endl;
//     bool done = false;

//     if (false) // some condition to check if the simulation is done
//     {
//         done = true;
//     }

//     return done;

// }
// void JamesCC::resetStepVariables()
// {
//     if (debug) cout << "\tJamesCC resetStepVariables()" << endl;
// }

// ObsType JamesCC::computeObservation(){
//     if (debug) cout << "\tJamesCC computeObservation()" << endl; 
//     return getRLState();

// }
// RewardType JamesCC::computeReward(){
//     if (debug) cout << "\tJamesCC computeReward()" << endl;
//     return getReward();
// }