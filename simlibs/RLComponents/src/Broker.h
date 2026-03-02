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

struct BrokerDetails{
  std::string rlId; // String ID returned into the dictionary of pthon step() function
  bool isReset; // Whether the next step is used as a RESET step, rather than normal step
  cMessage* endOfStep; // Msg used to notify end of step
  
  ObsType observation;
  float reward;
  bool done;
};

struct StepDetails{
  std::string rlId; // String ID returned into the dictionary of pthon step() function
  bool isReset; // Whether the next step is used as a RESET step, rather than normal step
  cMessage* stepMsg; // Msg used to notify end of step

  float stepSize; //Time in seconds (might delete)
};

class Broker : public cSimpleModule, public cListener
{
protected:
  bool allAgentsDone;
  // Map of active agents. Key is the string name of the agent (TODO: full path or just conn-N?), valiue is 
  // is a struct containing info on the current stepping logic
  std::unordered_map<std::string, BrokerDetails> activeAgents;
  std::unordered_map<std::string, StepDetails> activeAgentsStepper;

  using cListener::finish;
  virtual void finish() override;
  virtual void initialize() override;
  virtual void handleMessage(cMessage *msg) override;
  void receiveSignal(cComponent *source, simsignal_t signalID, cObject *value, cObject *obj) override;
  void receiveSignal(cComponent *source, simsignal_t signalID, const char *value, cObject *obj) override;

  simsignal_t actionResponse = registerSignal("actionResponse"); //communication signal from stepper to sender with the action from ray
  simsignal_t pullObservations = registerSignal("pullObservations"); //communication signal from stepper to sender requestiing the obsevations.

public:
  ~Broker();
 
  // Get the observation
  ObsType getObservation(std::string id);
  std::unordered_map<std::string, ObsType> getObservations();

  // Get the reward
  RewardType getReward(std::string id);
  std::unordered_map<std::string, RewardType> getRewards();

  // Get done
  bool getDone(std::string id);

  bool getAllDone();


  std::unordered_map<std::string, bool> getDones();

  bool areAllAgentsDone();

  // Set the action and whether this is a rest step or normal step. 
  // Values are written in the Broker instance. Broker sends a signal to each Agent 
  // specified in the map. Agent will use the given action in the coming step.
  void setActionAndMove(std::unordered_map<std::string, std::tuple<ActionType, bool>> &actionsAndMoves);
};

#endif