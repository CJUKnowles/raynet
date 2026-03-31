#ifndef OBSERVER_H_
#define OBSERVER_H_

#include "omnetpp/cobject.h"
#include "omnetpp/csimulation.h"
#include <string.h>
#include <omnetpp.h>
#include <unordered_map>
#include <deque>

using namespace omnetpp;
using namespace std;

// A single state representing the entire network, containing 
class GlobalState : public cObject, noncopyable
{
  public:
    // Global state (computed from many LocalStates)
    double minThroughput;   // Minimum current throughput of all flows
    double maxThroughput;   // Maximum current throughput of all flows
    double ovrThroughput;   // Overall current throughput of all flows (sum)
    double avgLatency;      // Average current latency of all flows
    double minCwnd;         // Minimum current cwnd of all flows
    double maxCwnd;         // Maximum current cwnd of all flows
    double avgCwnd;         // Average current cwnd of all flows
    double lossRatio;       // Average current loss ratio of all flows
    double numFlows;        // Current number of flows (agents)

    // Ground truth (network parameters)
    double LINK_DELAY;
    double BUFFER_SIZE;
    double BANDWIDTH;

    // Rewards
    double reward;
    double fairness;
    // etc..

    // Meta
    bool needsUpdating = false; // True if new localStates have been received, but GlobalState has yet to be updated
    void reset() {
      minThroughput = 9999999999;
      maxThroughput = 0;
      ovrThroughput = 0;
      avgLatency = 99999999;
      minCwnd = 999999999;
      maxCwnd = 0;
      avgCwnd = 0;
      lossRatio = 99999999;
      numFlows = 0;
      needsUpdating = false;
    }
  };

// A single state reported by an agent, containing a timestamp and several other parameters
class LocalState : public cObject, noncopyable
{
  public:
    simtime_t timestamp;
    double throughput;
    double maxThroughput;
    double delay;
    double minDelay;
    double cwnd;
    double lossRate;
    double inflight;

    LocalState() {
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
  // Values
  GlobalState* globalState; // The most recently computed global state
  std::unordered_map<std::string, StateHistory> astreaAgents; // Map of all Astrea agents. <agent_id, agent_current_info>

  // Omnet setup stuff
  virtual void initialize() override;
  using cListener::finish;
  virtual void finish() override;

  // Omnet signalling/scheduling stuff
  simsignal_t globalStateResponseSig = registerSignal("globalStateResponse");   // Signal used to report global state metrics to an agent upon request
  void receiveSignal(cComponent *source, simsignal_t signalID, cObject *value, cObject *obj) override;
  void receiveSignal(cComponent *source, simsignal_t signalID, const char *value, cObject *obj) override;

  // Observer
  void computeGlobalState(); // Updates the global state based on the most recently collected localStates
};

#endif