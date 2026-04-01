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

    // Ground truth (network parameters) // TODO: Set these dynamically
    double LINK_DELAY = .02;
    double BUFFER_SIZE = 1024;
    double BANDWIDTH = 6000;

    // Reward metrics
    double throughputMetric;
    double latencyMetric;
    double lossMetric;
    double fairnessMetric;
    double stabilityMetric;
    double reward;
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

    void printSummary() {
      cout << "\tGlobalState Summary: " << endl;
      cout << "\t\t" << "throughputMetric: " << throughputMetric << endl;
      cout << "\t\t" << "latencyMetric: " << latencyMetric << endl;
      cout << "\t\t" << "lossMetric: " << lossMetric << endl;
      cout << "\t\t" << "fairnessMetric: " << fairnessMetric << endl;
      cout << "\t\t" << "stabilityMetric: " << stabilityMetric << endl;
      cout << "\t\t" << "Reward: " << reward << endl;
    }
  };

// A single state reported by an agent, containing a timestamp and several other parameters
class LocalState : public cObject, noncopyable
{
  public:
    simtime_t timestamp;

    // Raw statistics
    double throughput;
    double latency;
    double cwnd;
    double lossRate;
    double inflight;

    // Observation values (mostly normalized, aside fomr the max/mins)
    double throughputRatio;
    double maxThroughput;
    double latencyRatio;
    double minLatency;
    double cwndRatio;
    double lossRateRatio;
    double inflightRatio;

    LocalState() {
    }

};

// A deque of the past n states reported by a given agent
struct StateHistory {
  std::deque<LocalState*> history;           // This agent's past n reported observations
  size_t max_history_length = 3;             // n, how many state entries should be stored for a given agent (just an unsigned int)
  double avgThroughput = 0;                  // Average throughput over entire history
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

  // Returns the average throughput of a given flow's entire history
  double getAverageThroughput() const {
    if (history.empty()) {
        return 0.0;
    }

    double throughputSum = 0.0;
    for (const LocalState* entry : history) {
        throughputSum += entry->throughput;
    }

    return throughputSum / static_cast<double>(history.size());
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
  // State
  GlobalState* globalState; // The most recently computed global state
  std::unordered_map<std::string, StateHistory> astreaAgents; // Map of all Astrea agents. <agent_id, agent_current_info>
  
  // Params
  double delayCoeff = 1.5; // Delays below minDelay*delayCoeff will be treated as optimal

  // Reward weights // TODO: Allow default weights to be overridden by the config.ini
  double throughputWeight = 0.1;
  double latencyWeight = 0.02;
  double lossWeight = 1.0;
  double fairnessWeight = 0.02;
  double stabilityWeight = 0.01;

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