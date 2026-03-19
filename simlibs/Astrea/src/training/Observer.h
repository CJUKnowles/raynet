#ifndef OBSERVER_H_
#define OBSERVER_H_

#include "omnetpp/cobject.h"
#include <string.h>
#include <omnetpp.h>
#include <unordered_map>
#include <deque>

using namespace omnetpp;
using namespace std;

class GlobalState : public cObject, noncopyable
{
  public:
    double average_throughput;
    double fairness;
    double something;
};

// A single state reported by an agent, containing a timestamp and several other parameters
class LocalState : public cObject, noncopyable
{
  public:
    simtime_t timestamp;
    double throughput;
    double srtt;

    LocalState(double throughput, double srtt) {
      this->throughput = throughput;
      this->srtt = srtt;
    }
};

// A deque of the past n states reported by a given agent
struct StateHistory {
  std::deque<LocalState*> history;           // This agent's past n reported observations
  size_t max_history_length = 3;             // n, how many state entries should be stored for a given agent (just an unsigned int)
  
  // Constructor - Maybe unnecessary
  StateHistory() {

  }

  // Destructor - deletes any remaining LocalState objects from memory
  ~StateHistory() {
    for (LocalState* entry : history) {
      delete entry;
    }
  }

  // Add a state to the history, while maintaining its max size and freeing memory as needed
  void addStateEntry(LocalState* entry) {
    history.push_front(entry);

    // Remove the oldest state (if necessary) and free its memory
    if (history.size() > max_history_length) {
      LocalState* stateToRemove = history.back();
      history.pop_back();
      delete(stateToRemove);
    }
  }
};

/*
  The Observer collects and maintains a history of states from agents.
  Upon request, the Observer uses these states to compute and return global state metrics, like fairness.
  Astrea agents use these global state metrics as rewards to emphasize global health of the network rather than individual performance.
*/
class Observer : public cSimpleModule, public cListener
{
protected:
  // Omnet setup stuff
  virtual void initialize() override;
  using cListener::finish;
  virtual void finish() override;

  // Map of all Astrea agents. <agent_id, agent_current_info>
  std::unordered_map<std::string, StateHistory> astreaAgents;

  // Omnet signalling/scheduling stuff
  simsignal_t globalStateResponseSig = registerSignal("globalStateResponse");   // Signal used to report global state metrics to an agent upon request
  void receiveSignal(cComponent *source, simsignal_t signalID, cObject *value, cObject *obj) override;
  void receiveSignal(cComponent *source, simsignal_t signalID, const char *value, cObject *obj) override;

  // Getters
  double computeAverageThroughput();
};

#endif