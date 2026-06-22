#include "omnetpp/csimulation.h"
#include <GymApi.h>
#include <string>


GymApi::GymApi()
    : env(nullptr), simulationPtr(nullptr), event(nullptr), bootconfigptr(nullptr), inifilePtr(nullptr)
{
}

void GymApi::cleanupmemory(){
    if (env)
        env->endSimulation();
    else if (simulationPtr)
        simulationPtr->deleteNetwork();

    cSimulation::setActiveSimulation(nullptr);
    delete simulationPtr;  // will delete app as well
    simulationPtr = nullptr;
    env = nullptr;
    
    // Below are manual cleanup steps to prevent memory leaks and dangling pointers.
    // Some of these are likely best practices and should be kept, but cause crashes during cleanup in new OMNeT++ versions.
    // Most of these should be added back in once the reset lifecycle listener/segfault bug is fixed, but for now they are commented out to prevent crashes during cleanup.
    CodeFragments::executeAll(CodeFragments::SHUTDOWN);
    
    // cout << "GymApi.cc::cleanupmemory(): Manually removing all dead lifecyclelisteners from static env:" << endl;
    // int size = getEnvir()->listeners.size();
    // for (int listener = 0; listener < size; listener++) {
    //     cout << "\tpopping..." << endl;
    //     getEnvir()->listeners.pop_back();
    // }
    
    // cout << "Done removing listeners. Listener list after removal: " << endl;
    // env is owned by cSimulation and was deleted with simulationPtr above.
    // delete simulationPtr;
    // delete event;
    // delete bootconfigptr;
    // delete inifilePtr;
}

// Generates a new OMNeT++ simulation and environment, and initializes them with the provided ini file and section name.
void GymApi::initialise(std::string _iniPath, std::string sectionName){
    if (env || simulationPtr)
        cleanupmemory();

    // initializations
    CodeFragments::executeAll(CodeFragments::STARTUP);
    SimTime::setScaleExp(-12);
    // char s1[] = "";
    std::vector<char*> cstrings; // final arguments for Omnet++
    // set up an environment for the simulation
    env = new Cmdrlenv();
    bootconfigptr = new SectionBasedConfiguration();
    inifilePtr = new InifileReader(); 
    //Read simulation configuration parameters from inifile
    inifilePtr->readFile(_iniPath.c_str());
    // activate [General] section so that we can read global settings from it
    bootconfigptr->setConfigurationReader(inifilePtr);
    //bootconfigptr->setActiveSection(sectionName.c_str());
    
    // Copy lifecycle listeners from the static environment to the simulation environment. Prevents missing INITSTAGE errors.
    for (auto l : getEnvir()->getLifecycleListeners()) {
        env->addLifecycleListener(l); 
    }

    // Set the simulation environment to be the active environment (instead of the static environment).
    simulationPtr = new cSimulation("simulation", env);
    cSimulation::setActiveSimulation(simulationPtr);

    // Finally, initialize the simulation environment and its components.
    env->initialiseEnvironment(cstrings.size(), &cstrings[0],bootconfigptr, sectionName);
}

// Returns the current OMNeT++ simulation time in seconds.
double GymApi::simTime(){
    cSimulation *simulation = getSimulation();
    if (!simulation)
        return 0.0;
    return simulation->getSimTime().dbl();
}

// Performs a single step without an action, and returns the resulting observations from the environment
 std::unordered_map<std::string, ObsType > GymApi::reset(){
    // Reset the environment
    bool isReset = true;
    std::unordered_map<std::string, ObsType > resetObs;
    std::string id = env->step(0, isReset);

    // Return early and notify the trainer if the simulation has concluded
    if(id == "SIMULATION_END"){
        env->endSimulation();
        ObsType obs;
        std::unordered_map<std::string, ObsType> obss = { {"SIMULATION_END", obs} };
        return obss;   
    }

    // Otherwise, return initial observations from the environment
    cModule *mod = getSimulation()->getModuleByPath((getSimulation()->getSystemModule()->getFullPath()+string(".broker")).c_str());
    Broker *target = check_and_cast<Broker *>(mod);
    auto obss = target->getObservations();
    int numObservationsCollected = target->invalidateOldStates();
    bool simDone = false;
    return obss;
}

// Progresses the simulation by calling env->step() (Cmderlenv::step()) with the provided action, and returns the resulting observations, rewards, and done flags for all agents in a tuple.
std::tuple< std::unordered_map<std::string, ObsType>,
            std::unordered_map<std::string, RewardType>,
            std::unordered_map<std::string,bool>,
            std::unordered_map<std::string,double>
            > GymApi::step(std::unordered_map<std::string, ActionType> actions){
    
    //Create container for return tuple of step method. 
    //Contains:
    //obs, reward, dones, info
    //where info is a numeric metadata dict containing simDone and simulation time.
    std::tuple<std::unordered_map<std::string, ObsType >, std::unordered_map<std::string, RewardType > , std::unordered_map<std::string,bool > , std::unordered_map<std::string,double > > returnTuple;
    bool isReset = false;
    std::string id = env->step(actions, isReset);
    
    // Collects obs, rewards, dones, etc. from the Broker and return to the trainer
    cModule *mod = getSimulation()->getModuleByPath(( getSimulation()->getSystemModule()->getFullPath()+ string(".broker")).c_str());
    Broker *target = check_and_cast<Broker *>(mod);
    auto obss = target->getObservations();
    auto rewards = target->getRewards();
    auto dones = target->getDones();
    int numObservationsCollected = target->invalidateOldStates();
    
    // Add extra done info to send to the trainer
    bool allDone = target->getAllDone();
    dones.insert({"__all__", allDone});
    bool simDone = false;
    if(id == "SIMULATION_END"){
        simDone = true;
        env->endSimulation();
    }

    returnTuple = {
        obss,
        rewards,
        dones,
        { {"simDone", simDone ? 1.0 : 0.0}, {"time_s", simTime()} }
    };
    return returnTuple;
}


void GymApi::shutdown(){    
    env->endSimulation();
}
  
