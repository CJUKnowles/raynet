#include "omnetpp/clistener.h"
#include "omnetpp/simkerneldefs.h"
#ifndef __Astraea_CC_H_
#define __Astraea_CC_H_

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
#include <transportlayer/tcp/TcpPacedConnection.h>
#include <transportlayer/tcp/flavours/TcpPacedNoCC.h>

using namespace omnetpp;
using namespace inet::tcp;
using namespace inet;
using namespace learning;

class Astraea : public TcpPacedNoCC, public RLInterface
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
    Astraea();
    virtual ~Astraea();

    // 
    using RLInterface::receiveSignal;

    // TcpCubic Overrides (These are mostly unchanged, and just used to gather statistic or disable automatic pacing)
    virtual void rttMeasurementComplete(simtime_t tSent, simtime_t tAcked) override;
    virtual void receivedDataAck(uint32_t firstSeqAcked) override;
    virtual void receivedDuplicateAck() override;
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
    bool debug = false; // Prints debug messages if true
    bool takeActions = true; // Skips Astraea actions if false

    // Astraea parameters (Default values here, overridden in Astraea.ini)
    double actionControlCoeff = 0.025; // Alpha term used by the original Astraea action helper.
    double fixedIntervalDuration=0.03;  // Seconds between steps

    // Raw interval metrics returned to Python for learner-specific processing.
    double astraeaThroughput = 0.0;
    double astraeaMaxThroughput = 0.0;
    double astraeaDelaySum = 0.0;
    double astraeaMinDelay = 0.0;
    double astraeaLossRate = 0.0;
    double deliveryRateSampleSum = 0.0;
    uint32_t deliveryRateSampleCount = 0;

    // Astraea helper variables.
    simtime_t lastIntervalTime = 0.0;
    uint32_t rttReportCount = 0;
    double bytesDelivered = 0.0;
    double bytesLost = 0.0;
    double last_bytes_delivered = 0.0;
    double last_bytes_lost = 0.0;
    bool pendingIntervalReset = false;

private:
    void recordDeliveryRateSample();
  };
#endif
