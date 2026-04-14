/*
 * Broker.h
 *
 *  Created on: Oct 2, 2021
 *      Author: basil
 */

#ifndef BROKER_H_
#define BROKER_H_

#include <string.h>
#include <iostream>
#include <array>
#include <omnetpp.h>
#include "BrokerData.h"
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include "cobjects.h"
#include "typedefs.h"

#include "omnetpp/cobject.h"
#include "omnetpp/csimulation.h"

using namespace omnetpp;
using namespace std;

/*
  Contains some stepping data for a given agent.
  Mostly passed between the agent (RLInterface instance) and broker to facilitate communication.
*/
struct BrokerDetails{
  // Agent info
  std::string rlId;     // String ID returned into the dictionary of pthon step() function
  cMessage* STEPmsg;    // Msg used to notify an agent to compute observation/reward and pass it to the broker
  bool done;

  // Most recent obs info
  ObsType observation;        // The observation value itself, usually a tuple of floats or doubles
  bool isReset;               // Whether the next step is used as a RESET step, rather than normal step
  float reward;               // The reward value itself
  bool uncollected = false;   // True if current state (obs/reward/done) is new but not yet collected by trainer. Set to false with invalidateOldStates().
};

/*
  The broker facilitaties communications between the agents and the upper layer of RayNet (Trainer, rllib, etc.).
  It is responsible for collection and forwarding observations/rewards/actions,
  as well as scheduling STEP and EOS (end-of-step) events for agents.
*/
class Broker : public cSimpleModule, public cListener
{
protected:
  // Omnet setup stuff
  virtual void initialize() override;
  using cListener::finish;
  virtual void finish() override;

  // Map of all agents. <agent_id, agent_current_info>
  std::unordered_map<std::string, BrokerDetails> activeAgents;
  bool allAgentsDone = false;

  // Omnet signalling/scheduling stuff
  cMessage* EOSmsg = new cMessage((std::string("EOS")).c_str());    // Event message signal an end-of-step, in which RayNet collects uncollected observations from the Broker.
  virtual void handleMessage(cMessage *msg) override;                       // Intercepts STEP events to request agent observations
  simsignal_t obsRequestSig = registerSignal("obsRequest");           // Signal used to request observations from agents
  simsignal_t performActionSig = registerSignal("performAction");     // Signal used to forward actions to agents
  void receiveSignal(cComponent *source, simsignal_t signalID, cObject *value, cObject *obj) override;
  void receiveSignal(cComponent *source, simsignal_t signalID, const char *value, cObject *obj) override;
public:
  // Forwards the provided actions to every agent in the map.
  void setActionAndMove(std::unordered_map<std::string, std::tuple<ActionType, bool>> &actionsAndMoves);

  // Various getters
  ObsType getObservation(std::string id);
  std::unordered_map<std::string, ObsType> getObservations();
  RewardType getReward(std::string id);
  std::unordered_map<std::string, RewardType> getRewards();
  int invalidateOldStates();
  bool getDone(std::string id);
  std::unordered_map<std::string, bool> getDones();
  bool getAllDone();
  bool areAllAgentsDone();
  bool areAllObsUncollected();

  // Parameters
  enum ObsCollectionMode {
    IMMEDIATE,    // EOS events are scheduled immediately after every STEP event. Each step returns a single-entry observation dict like: {agent27: <1.1, 1.2, 3.1, 4>}
    GROUPED,      // EOS events are scheduled only when all activeAgents have fresh observations. Good for agents that step simultaneously. Each observation dict contains entries for ALL active agents.
    INTERVALED    // EOS events are scheduled at a configurable time interval. TODO: implement this lol
  };
  enum ObsCollectionMode obsCollectionMode;
};

#endif