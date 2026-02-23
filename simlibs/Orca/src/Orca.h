#ifdef ORCA
#ifndef __ORCA_CC_H_
#define __ORCA_CC_H_

#include <omnetpp.h>
#include <iostream>
#include <string>
#include <math.h>
#include <array>
#include <random>
#include <tuple>
#include "BrokerData.h"
#include "RLInterface.h"
#include <inet/common/INETDefs.h>

#include "MonitorInterval.h"
#include "inet/transportlayer/tcp/flavours/TcpNewReno.h"
#include <inet/transportlayer/tcp/Tcp.h>
#include <inet/transportlayer/tcp/TcpConnection.h>
#include <transportlayer/tcp/flavours/TcpCubic.h>

using namespace omnetpp;
using namespace inet::tcp;
using namespace inet;
using namespace learning;

class Orca : public TcpCubic, public RLInterface
{
protected:
    // am I running on active open (client) or passive open connection (server)
    bool isActive;
    // OMNeT++ message used to schedule monitor intervals
    cMessage *RLStep;

    // utility object that keeps track of Monitor Intervals and relevant calculations
    MonitorIntervalsHandler miHandler;
    //Signals for result recording
    simsignal_t throughputSignal;
    simsignal_t actionSignal;
    simsignal_t dupAcksSignal;
    simsignal_t rttGradientSignal;
    simsignal_t tickSignal;
    simsignal_t miQueueSizeSignal;
    uint32_t dupAcks;
public: // General use
    Orca();
    virtual ~Orca();

    // TcpCubic Overrides (Congestion control functions we want to alter the behaviour of, or grab statistics with)
    virtual void receivedDataAck(uint32_t firstSeqAcked) override;
    virtual void receivedDuplicateAck() override;
    virtual void established(bool active) override; // Called when the TCP CONNECTION is established (some time AFTER startup!)
    virtual void rttMeasurementComplete(simtime_t tSent, simtime_t tAcked) override;  // Overridden so we can track 

    // RLInterface Overrides (virtual functions that must be overridden)
    virtual void initialize() override; // This also overrides the TcpNewReno initialize(). Be sure to super() both of them.
    virtual void resetStepVariables()override;
    virtual void decisionMade(ActionType action) override; // Call back from RLInterface. Called when the action from the agent has been received.
    virtual ObsType computeObservation()override;
    virtual RewardType computeReward()override;
    virtual ObsType getRLState() override;
    virtual RewardType getReward() override;
    virtual bool getDone() override;
    virtual void cleanup() override;

    // RL-related and utility variables
    double RLStepInterval = 1.0; // How many sim seconds to wait between RL steps, 1 by default. Ideally I will modify the step size based on RTT instead of using this.
    int RLStepsTaken = 0; // How many RLSteps have been completed so far.
    int maxRLSteps = 10000; // How many training steps should be taken before this agent reports itself as done.
    bool debug = false; // Prints debug messages if true

    // Orca parameters
    double delayWeight = 1; // Beta term from Orca paper. Delay only degrades reward if RTT > baseRTT*delayWeight. Larger values emphasize throughput in reward calculation.

    // Orca observation values (These will be updated over time by TCP functions, returned as observations, then reset. Rinse and repeat.)
    double orcaThroughput=0.0;    // The average delivery rate (throughput) over the last interval
    double orcaLossRate=0.0;      // The average loss rate of packets over the last interval
    double orcaDelay=0.0;         // The average delay of packets over the last interval
    double orcaACKTotal=0.0;      // The number of valid acknowledgements over the last interval
    double orcaIntervalDuration=0.0;  // The simtime elapsed over the last interval
    double orcaSRTT=0.0;          // The smoothed RTT of (all?) packets so far
    double orcaCwnd=0.0;          // The current congestion window (don't really need a new variable here, this is just useful for reference. Just use conn->snd_cwnd)
    double orcaMaxThroughput=1.0; // The maximum delivery rate so far
    double orcaMinDelay=9999;      // The minimum packet delay so far. Initialize to large value so the minimum is guaranteed to update.

    // Orca helper variables (mostly used to facilitate computing the observations)
    simtime_t lastIntervalTime = 0.0;
    double lastIntervalSentBytes = 0.0; // Whatever value state->sentBytes returned last interval. The TOTAL so far; NOT what was sent DURING the last interval.
    uint32_t lastIntervalSndUna = 0;  // Whatever the oldest reported unACK'd byte was at the last monitor interval
    uint32_t bytesSentTotal = 0;
    uint32_t rttReportCount = 0;    // How many rtt reports we received this interval
    // Old - to be removed
    double lastStepCwnd=0.0; // What the CWND was at the end of the last step 
    double lastStepDelay=0.0;  // What the delay was at the end of the last stp
    double lastStepSent=0.0;    // How many packets were sent during the last step
    double lastStepSSThresh=0.0; // What was SSthresh last step
    double slowstartMultiplier=1; // The RL action: changes how quickly slow start increases CWND
    double maxCwnd=1.0; // The max cwnd observed in an interval
    double maxACKTotal=1.0; // The max ACK total observed in an interval
  };
#endif
#endif