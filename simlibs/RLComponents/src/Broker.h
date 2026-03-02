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

using namespace omnetpp;
using namespace std;

/*
  Contains some stepping data for a given agent.
  Mostly passed between the agent (RLInterface instance) and broker to facilitate communication.
*/
struct BrokerDetails{
  // Unique IDs and event names for this agent
  std::string rlId;     // String ID returned into the dictionary of pthon step() function
  cMessage* endOfStep;  // Msg used to notify end of step
  cMessage* stepMsg;    // Msg used to notify end of step

  // Current state
  bool isReset;         // Whether the next step is used as a RESET step, rather than normal step
  ObsType observation;
  float reward;
  bool done;
  float stepSize; //  Time between steps in seconds (might delete)
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
  virtual void handleMessage(cMessage *msg) override; // Intercepts STEP events to request agent observations
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
  bool getDone(std::string id);
  std::unordered_map<std::string, bool> getDones();
  bool getAllDone();
  bool areAllAgentsDone();
};

#endif