#include "omnetpp/clistener.h"
#include "omnetpp/simkerneldefs.h"
#ifdef ASTREA
#ifndef __ASTREA_CC_H_
#define __ASTREA_CC_H_

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
#include <transportlayer/tcp/flavours/TcpPacedNoCC.h>

using namespace omnetpp;
using namespace inet::tcp;
using namespace inet;
using namespace learning;

class Astrea : public TcpPacedNoCC, public RLInterface
{
protected:
    // am I running on active open (client) or passive open connection (server)
    bool isActive;

    //Signals for result recording
    simsignal_t throughputSignal;
    simsignal_t actionSignal;
    simsignal_t srttSignal;
    simsignal_t cwndSignal;
    simsignal_t intervalDurationSignal;
    uint32_t dupAcks;
public: // General use
    Astrea();
    virtual ~Astrea();

    // 
    using RLInterface::receiveSignal;
    virtual void receiveSignal(cComponent *source, simsignal_t signalID, double value, cObject *details) override 
    {
      std::string agentId = ((cString*) details)->str;
      if (agentId != this->stringId) {
        //if (debug) cout << "\t\t" << stringId << " recieved signal meant for " << agentId << ", ignoring" << endl;
        return;
      }

      const char *signalName = inet::tcp::Tcp::getSignalName(signalID);
      if (strcmp(signalName, "globalStateResponse") == 0) {
        this->reward = value;
      } else {
        cout << "Received invalid signal" << endl;
      }
    }

    // TcpCubic Overrides (These are mostly unchanged, and just used to gather statistic or disable automatic pacing)
    virtual void established(bool active) override; // Called when the TCP CONNECTION is established (some time AFTER startup!)

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
    double initialStepLength = 1.0; // How many simseconds to wait before scheduling the initial step. SRTT will be used for future step lengths.
    int RLStepsTaken = 0; // How many RLSteps have been completed so far.
    int maxRLSteps = 10000; // How many training steps should be taken before this agent reports itself as done.
    bool debug = false; // Prints debug messages if true
    bool takeActions = true; // Skips Astrea actions if false

    // Astrea parameters (Default values here, overridden in astrea.ini)
    double rewardDelayForgiveness = 1; // 
    double rewardLossMultiplier = 1;   // 
    double actionControlCoeff = .025; // Alpha term from Astrea paper. Larger values allow larger changes to cwnd.
    double fixedIntervalDuration=0.03;  // Seconds between steps

    // Astrea observation values (These will be updated over time by TCP functions, returned as observations, then reset. Rinse and repeat.)
    double astreaThroughput=0.0;    // The average delivery rate (throughput) over the last interval
    double astreaLossRate=0.0;      // The average loss rate of packets over the last interval
    double astreaACKTotal=0.0;      // The number of valid acknowledgements over the last interval
    double astreaSRTT=0.0;          // The smoothed RTT of (all?) packets so far
    double astreaCwnd=0.0;          // The current congestion window (don't really need a new variable here, this is just useful for reference. Just use conn->snd_cwnd)
    double astreaMaxThroughput=0; // The maximum delivery rate so far
    double astreaMinDelay=9999;      // The minimum packet delay so far. Initialize to large value so the minimum is guaranteed to update.
    double astreaPaceRate=1;        // Bytes sent per second. Usually smaller than cwnd.
    double astreaDelayMetric=1;     // A measure of how close the currenty delay is to optimal. Will be 1 as long as the delay is within the forgiveness window.

    // Astrea helper variables (mostly used to facilitate computing the observations)
    simtime_t lastIntervalTime = 0.0;
    double last_snd_max = 0.0; // Whatever value state->snd_max returned last interval. The TOTAL so far; NOT what was sent DURING the last interval.
    uint32_t last_snd_una = 0;  // Whatever the oldest reported unACK'd byte was at the last monitor interval
    uint32_t last_rexmit_count = 0; // How many bytes were retransmitted in TOTAL, as reported last interval
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
    double retransmissionRate; // The most recent measurement of bytes retransmitted.
    bool first_slowstart_complete = false; // Do not take astrea actions until the first slow start phase has completed. This allows the initial state (max througphut and min delay) to form naturally and prevents deadlocks.
    
    // Observer signals
    simsignal_t registerAstreaAgentSig = owner->registerSignal("registerAstreaAgent");
    simsignal_t astreaStateReportSig = owner->registerSignal("astreaStateReport");
    simsignal_t globalStateRequestSig = owner->registerSignal("globalStateRequest");

    double reward = 0; // Will automatically be set when globalStateResponse signal is received
  };
#endif
#endif