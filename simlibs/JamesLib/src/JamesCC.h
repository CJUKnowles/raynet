#ifndef __JAMES_CC_H_
#define __JAMES_CC_H_

#include <omnetpp.h>
#include <iostream>
#include <string>
#include <math.h>
#include <array>
#include <random>
#include <tuple>
#include "BrokerData.h"
#include "RLInterface.h"

#include "MonitorInterval.h"
#include "inet/transportlayer/tcp/flavours/TcpNewReno.h"
#include <inet/transportlayer/tcp/Tcp.h>
#include <inet/transportlayer/tcp/TcpConnection.h>

using namespace omnetpp;
using namespace inet::tcp;
using namespace inet;
using namespace learning;

class JamesCC : public TcpNewReno, public RLInterface
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
    JamesCC();
    virtual ~JamesCC();

    virtual void initialize() override; // This is either overriding a deprecated function in RLInterface, or actually overriding something from TcpNewReno. Look into it.
    // RLInterface Overrides (virtual functions that must be overridden)
    virtual void cleanup() override;
    virtual void decisionMade(ActionType action) override; // Call back from RLInterface. Called when the action from the agent has been received.
    virtual ObsType getRLState() override;
    virtual RewardType getReward() override;
    virtual bool getDone() override;
    virtual void resetStepVariables()override;
    virtual ObsType computeObservation()override;
    virtual RewardType computeReward()override;

    // TcpNewReno Overrides
    virtual void receivedDataAck(uint32_t firstSeqAcked) override;
    virtual void receivedDuplicateAck() override;
    virtual void recalculateSlowStartThreshold() override;
    virtual void processRexmitTimer(TcpEventCode &event) override;
    virtual void established(bool active) override;
    // virtual void processTimer(cMessage *timer, TcpEventCode &event) override; // Used to intercept self-scheduled events, like the RL step

    // Custom, for this protocol
    double RLStepInterval = 1.0; // How many sim seconds to wait between RL steps, 1 by default. Ideally I will modify the step size based on RTT instead of using this.
    void getObservationVec(std::vector<double> &obs); 
    int RLStepsTaken = 0; // How many RLSteps have been completed so far.
    bool debug = false; // Prints debug messages if true

  };
#endif