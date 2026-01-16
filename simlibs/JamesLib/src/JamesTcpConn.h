#ifndef TRANSPORTLAYER_JAMETCPCONN_H_
#define TRANSPORTLAYER_JAMETCPCONN_H_

#include "RLInterface.h"
#include <inet/transportlayer/tcp/Tcp.h>
#include <inet/transportlayer/tcp/TcpConnection.h>

using namespace inet::tcp;
using namespace omnetpp;
using namespace learning;
/*
 * Overrides Tcp implementation to define new NED parameters.
 */
class JamesTcpConn : public TcpConnection, public RLInterface
{
public:
    JamesTcpConn();
    virtual ~JamesTcpConn();

    // RLInterface Overrides
    virtual void initialize() override;
    virtual void cleanup() override;
    virtual void decisionMade(ActionType action) override; // Call back from RLInterface. Called when the action from the agent has been received.
    virtual ObsType getRLState() override;
    virtual RewardType getReward() override;
    virtual bool getDone() override;
    virtual void resetStepVariables()override;
    virtual ObsType computeObservation()override;
    virtual RewardType computeReward()override;
    


    // Mine!
    ObsType state; // array declared
    cMessage* initMsg; // Msg used to notify end of step
    bool isRegistered;
    int stepInterval; // how many seconds to wait between steps

    cMessage *monitorInterval;
    // omnet overrides
    virtual void handleMessage(cMessage *msg) override;
   // virtual void processTimer(cMessage *timer, TcpEventCode &event) override; // Doesnt exist for TcpConnection. Going to try extending TcpNewReno

    
};

#endif /* TRANSPORTLAYER_RLTCP_H_ */
