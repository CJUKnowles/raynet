#include "omnetpp/clistener.h"
#include "omnetpp/simkerneldefs.h"
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
#include <numeric>

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
public:
    // true if this is the client, false if server
    bool isActive;

    //Signals for result recording
    simsignal_t throughputSignal;
    simsignal_t actionSignal;

public: // General use
    Orca();
    virtual ~Orca();

    // 
    using RLInterface::receiveSignal;
    virtual void receiveSignal(cComponent *source, simsignal_t signalID, double value, cObject *details) override 
    {
      TcpPacedConnection* pacedConn = dynamic_cast<TcpPacedConnection*>(conn);
      if (signalID == pacedConn->retransmissionRateSignal) {
        retransmissionRate = value/8.0; // Retransmiitted bytes/s this interval
      }
    }

    // TcpCubic Overrides (These are mostly unchanged, and just used to gather statistic or disable automatic pacing)
    virtual void rttMeasurementComplete(simtime_t tSent, simtime_t tAcked) override;  // Used to track rtt-related stats for observations
    virtual void receivedDataAck(uint32_t firstSeqAcked) override;
    virtual void established(bool active) override; // Called when the TCP CONNECTION is established (some time AFTER startup!)
    virtual void receivedDuplicateAck() override; // Remove pacing rate change

    // RLInterface Overrides (Required by the RL agent)
    virtual void initialize() override; // This also overrides the TcpNewReno initialize(). Be sure to super() both of them.
    virtual void resetStepVariables()override;
    virtual void decisionMade(ActionType action) override; // Call back from RLInterface. Called when the action from the agent has been received.
    virtual std::optional<ObsType> computeObservation()override;
    virtual RewardType computeReward()override;
    virtual ObsType getRLState() override;
    virtual RewardType getReward() override;
    virtual bool getDone() override;
    virtual void cleanup() override;

    // RL-related and utility variables
    int RLStepsTaken = 0; // How many RLSteps have been completed so far.
    int maxRLSteps = 10000; // How many training steps should be taken before this agent reports itself as done.
    bool debug = false; // Prints debug messages if true
    bool takeActions = true; // Skips Orca actions if false

    // Orca parameters (Default values here, overridden in orca.ini)
    double rewardDelayForgiveness = 1; // Beta term from Orca paper. Delay only degrades reward if RTT > baseRTT*rewardDelayForgiveness. Larger values emphasize aggressive throughputs by forgiving delay increases.
    double rewardLossMultiplier = 1;   // Zeta term from Orca paper. Throughput is substracted by lossRate*rewardLossMultiplier in reward computation.  Larger values emphasize conservative throughputs by punishing loss. 
    
    // Orca observation values (These will be updated over time by TCP functions, returned as observations, then reset. Rinse and repeat.)
    double orcaThroughput=0.0;    // The average delivery rate (throughput) over the last interval
    double orcaLossRate=0.0;      // The average loss rate of packets over the last interval
    double orcaDelay=0.0;         // The average delay of packets over the last interval
    double orcaDelaySum=0.0;      // Used to hold the current sum of reported delays over a given interval. Used to compute an average at the end.
    double orcaACKTotal=0.0;      // The number of valid acknowledgements over the last interval
    double fixedIntervalDuration;  // The simtime elapsed over the last interval
    double orcaSRTT=0.0;          // The smoothed RTT of (all?) packets so far
    double orcaCwnd=0.0;          // The current congestion window (don't really need a new variable here, this is just useful for reference. Just use conn->snd_cwnd)
    double orcaMaxThroughput=0.0; // The maximum delivery rate so far
    double orcaMinDelay=9999;      // The minimum packet delay so far. Initialize to large value so the minimum is guaranteed to update.
    double orcaPaceRate=1;        // Bytes sent per second. Usually smaller than cwnd.
    double orcaDelayMetric=1;     // A measure of how close the currenty delay is to optimal. Will be 1 as long as the delay is within the forgiveness window.

    // Orca helper variables (mostly used to facilitate computing the observations)
    simtime_t lastIntervalTime = 0.0;

    // Changing state variables (store deltas as doubles to prevent constant casting lol)
    double delta_snd_max;
    double delta_snd_una;
    double delta_ack_cnt;
    uint32_t last_snd_max = 0.0; // Whatever value state->snd_max returned last interval. The TOTAL so far; NOT what was sent DURING the last interval.
    uint32_t last_snd_una = 0;  // Whatever the oldest reported unACK'd byte was at the last monitor interval
    uint32_t last_ack_cnt = 0;

    uint32_t bytesSentTotal = 0;
    uint32_t rttReportCount = 0;    // How many rtt reports we received this interval
    // Old - to be removed
    double lastStepCwnd=0.0; // What the CWND was at the end of the last step 
    double lastStepDelay=0.0;  // What the delay was at the end of the last stp
    double lastStepSent=0.0;    // How many packets were sent during the last step
    double lastStepSSThresh=0.0; // What was SSthresh last step
    double slowstartMultiplier=1; // The RL action: changes how quickly slow start increases CWND
    double retransmissionRate=0.0; // The most recent measurement of bytes retransmitted.
    bool first_slowstart_complete = false; // Do not take orca actions until the first slow start phase has completed. This allows the initial state (max througphut and min delay) to form naturally and prevents deadlocks.
    bool fixedIntervals; // Dynamically adjust the interval duration to the SRTT
    double bytesDelivered=0.0; // Sum of all bytes delivered this interval
  };
#endif
#endif