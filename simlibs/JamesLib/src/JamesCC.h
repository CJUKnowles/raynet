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

using namespace inet::tcp;
using namespace omnetpp;
using namespace std;
namespace learning {
/**
 * TODO - Generated class
 */
 // Used to import cSimpleModule, now only imports TcpReno
class JamesCC : public TcpNewReno, public RLInterface
{
public: // General use
    JamesCC();
    virtual ~JamesCC();
    double high[4];
    ActionType action[2] = {0, 1};
    int steps_beyond_done;
    int steps;
    ObsType state; // array declared
    cMessage* initMsg; // Msg used to notify end of step

    bool isRegistered;
public: // Learning
    ObsType random();
    virtual void initialize();
    void step(ActionType action);
    virtual void handleMessage(cMessage *msg);
    virtual void finish();
    virtual void cleanup();
    virtual void decisionMade(ActionType action); // defines what to do when decision is made
    virtual ObsType  getRLState();
    virtual RewardType getReward();
    virtual bool getDone();
    virtual void resetStepVariables();
    virtual ObsType computeObservation();
    virtual RewardType computeReward();
    bool isConnectionPaced;
    uint32_t maxLearnWindow;
protected: // Signals (for results)
  MonitorIntervalsHandler miHandler;
  cMessage *monitorInterval;
    simsignal_t throughputSignal;
    simsignal_t actionSignal;
    simsignal_t dupAcksSignal;
    simsignal_t rttGradientSignal;
    simsignal_t tickSignal;
    simsignal_t miQueueSizeSignal;

};
#endif
}