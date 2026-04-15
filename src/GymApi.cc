#include "omnetpp/csimulation.h"
#include <GymApi.h>
#include <string>


GymApi::GymApi(){
}

void GymApi::cleanupmemory(){
    getSimulation()->deleteNetwork();
    cSimulation::setActiveSimulation(nullptr);
    delete simulationPtr;  // will delete app as well

    // CodeFragments::executeAll(CodeFragments::SHUTDOWN);
    
    // cout << "GymApi.cc::cleanupmemory(): Manually removing all dead lifecyclelisteners from static env:" << endl;
    // int size = getEnvir()->listeners.size();
    // for (int listener = 0; listener < size; listener++) {
    //     cout << "\tpopping..." << endl;
    //     getEnvir()->listeners.pop_back();
    // }
    
    // cout << "Done removing listeners. Listener list after removal: " << endl;

    // delete env;

    // delete simulationPtr;
    // delete event;
    // delete bootconfigptr;
    // delete inifilePtr;
}

void GymApi::initialise(std::string _iniPath, std::string sectionName){
    cStaticFlag dummy;
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
    
    for (auto l : getEnvir()->getLifecycleListeners()) {
        // Idea: rather than calling the staticEnvir that initially loads up, you may be calling the previous environment
        env->addLifecycleListener(l); // Called "app" in omnetpp's original code. This is a reference to cmdenv, qtenv, or in this case, cmrlenv.
    }
    
    simulationPtr = new cSimulation("simulation", env);
    cSimulation::setActiveSimulation(simulationPtr);
    
    env->initialiseEnvironment(cstrings.size(), &cstrings[0],bootconfigptr, sectionName);
}


 std::unordered_map<std::string, ObsType > GymApi::reset(){
    // Reset the environment
    bool isReset = true;
    std::unordered_map<std::string, ObsType > resetObs;

    // run the simulation

    std::string id = env->step(0, isReset);


    if(id != "SIMULATION_END"){
        cModule *mod = getSimulation()->getModuleByPath((getSimulation()->getSystemModule()->getFullPath()+string(".broker")).c_str());
        Broker *target = check_and_cast<Broker *>(mod);


        auto obss = target->getObservations();
        int numObservationsCollected = target->invalidateOldStates();
        bool simDone = false;
        // Don't do this anymore. getObservations() now returns only new observations. Set Broker.ObsCollectionMode to IMMEDIATE if you only want one obs per dict
        // // Prune any observations not from the agent that triggered this EOS
        // auto it = obss.begin();
        // while (it != obss.end()) {
        //     // Check if key's first character is F
        //     if (it->first != id) {
        //         // erase() function returns the iterator of the next
        //         // to last deleted element.
        //         it = obss.erase(it);
        //     } else
        //         it++;
        // }
        return obss;
    }
    else{
        ObsType obs;
        std::unordered_map<std::string, ObsType> obss = { {"SIMULATION_END", obs} };
        return obss;
    }
    
}

// std::tuple<std::unordered_map<std::string, ObsType >, std::unordered_map<std::string, RewardType > , std::unordered_map<std::string,bool > > GymApi::step(ActionType action){
    
//     std::tuple<std::unordered_map<std::string, ObsType >, std::unordered_map<std::string, RewardType > , std::unordered_map<std::string,bool > > returnTuple;
//     bool isReset = false;

//     string networkname("simplenetwork");
//     // We call step on the environment
//     env->step(action, isReset, networkname);


//     cModule *mod = getSimulation()->getModuleByPath((networkname+string(".broker")).c_str());
//     Broker *target = check_and_cast<Broker *>(mod);
    
//     returnTuple = {target->getObservations(), target->getRewards(), target->getDones()};

//     return returnTuple;
// }

std::tuple< std::unordered_map<std::string, ObsType>,
            std::unordered_map<std::string, RewardType>,
            std::unordered_map<std::string,bool>,
            std::unordered_map<std::string,bool>
            > GymApi::step(std::unordered_map<std::string, ActionType> actions){
    
    //Create container for return tuple of step method. 
    //Contains:
    //obs, reward, dones, info
    //where info is a dict (unordered_map) with a single key,value pair "simDone" to denote whether the simulation has been completed.
    std::tuple<std::unordered_map<std::string, ObsType >, std::unordered_map<std::string, RewardType > , std::unordered_map<std::string,bool > , std::unordered_map<std::string,bool > > returnTuple;
    bool isReset = false;
    std::string id = env->step(actions, isReset);

    cModule *mod = getSimulation()->getModuleByPath(( getSimulation()->getSystemModule()->getFullPath()+ string(".broker")).c_str());
    Broker *target = check_and_cast<Broker *>(mod);

    auto obss = target->getObservations();
    auto rewards = target->getRewards();
    auto dones = target->getDones();
    int numObservationsCollected = target->invalidateOldStates();
    bool allDone = target->getAllDone();
    dones.insert({"__all__", allDone});

    bool simDone = false;

    if(id == "SIMULATION_END"){
        simDone = true;
    }
    
    returnTuple = { obss, rewards, dones, { {"simDone", simDone} } };
    return returnTuple;
}


void GymApi::shutdown(){
    
    env->endSimulation();
}
  
