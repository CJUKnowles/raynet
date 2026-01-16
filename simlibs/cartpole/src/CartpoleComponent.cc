

#ifdef CARTPOLE

#include "CartpoleComponent.h"


Define_Module(CartpoleComponent);

CartpoleComponent::~CartpoleComponent(){
    cancelAndDelete(initMsg);
}

void CartpoleComponent::initialize()
{
    steps=0;	
    gravity = 9.8;
    masscart = 1.0;
    masspole = 0.1;
    total_mass = masspole + masscart;
    length = 0.5; // actually half the pole's length
    polemass_length = masspole * length;
    force_mag = 10.0;
    tau = 0.02; // seconds between state updates
    kinematics_integrator = "euler";

    // Angle at which to fail the episode
    theta_threshold_radians = 12 * 2 * M_PI / 360;
    x_threshold = 2.4;

    // Angle limit set to 2 * theta_threshold_radians so failing observation
    // is still within bounds.
    high[0] = x_threshold * 2;
    high[1] = 3.4028235e+38;
    high[2] = theta_threshold_radians * 2;
    high[3] = 3.4028235e+38;

    steps_beyond_done = -10;
    a = 2;
    b = 13;    

    state = random();

    isRegistered = false;

    initMsg = new cMessage("CARTPOLE-INIT"); 
    scheduleAt(simTime() + 1, initMsg);
}

void CartpoleComponent::handleMessage(cMessage *msg)
{
    if(msg->isSelfMessage()){
        scheduleAt(simTime() + 1, initMsg);
    
        if(!isRegistered){
            isRegistered = true;
            cObject* simtime = new cSimTime(1);
            this->setOwner(this);
            RLInterface::initialise();

            // Generate ID for this agent
            std::string s("cartpole");
            this->setStringId(s);

            emit(this->registerSig, stringId.c_str(), simtime); // could be missing this?
        }
    }
    else{
        EV_DEBUG << "Stepper should only receive self messages!" << std::endl;
    }
}

ObsType CartpoleComponent::random()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.05, 0.05);
    ObsType values;
    for (int n = 0; n < 4; ++n)
        values[n] = dis(gen);
    return values;
}

void CartpoleComponent::step(ActionType action)
{
    double x = state[0];
    double x_dot = state[1];
    double theta = state[2];
    double theta_dot = state[3];
    double force;
    steps++;
    if (action == 1)
    {
        force = force_mag;
    }
    else
    {
        force = force_mag * -1;
    }

    double costheta = cos(theta);
    double sintheta = sin(theta);

    double temp = (force + polemass_length * pow(theta_dot, 2) * sintheta) / total_mass;

    double thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * pow(costheta, 2) / total_mass));

    double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    // cout << "temp: " << temp << " thetaacc: " << thetaacc << " xacc: " << xacc <<endl;

    if (kinematics_integrator == "euler")
    {
        x = x + tau * x_dot;
        x_dot = x_dot + tau * xacc;
        theta = theta + tau * theta_dot;
        theta_dot = theta_dot + tau * thetaacc;
    }
    else
    {
        x_dot = x_dot + tau * xacc;
        x = x + tau * x_dot;
        theta_dot = theta_dot + tau * thetaacc;
        theta = theta + tau * theta_dot;
    }

    state = {x, x_dot, theta, theta_dot};

}

void CartpoleComponent::cleanup(){}

// Pre-step; 
void  CartpoleComponent::decisionMade(ActionType action){
    if(this->isReset){
        //Reset state for next iterations
        // steps_beyond_done = -10;
        // state = random();
        // std::cout << "Environment reset!" << std::endl;
        step(action);
    }else{
        step(action);
    }

}

// defines what to do when decision is made
ObsType CartpoleComponent::getRLState(){
    return state;
}

// Return a reward for every timestep the pole has not fallen
RewardType CartpoleComponent::getReward(){
    RewardType reward;
    if (done == false)
    {
        reward = 1.0;
    }
    else if (steps_beyond_done == -10)
    {
        // Pole just fell
        steps_beyond_done = 0;
        reward = 1.0;
    }
    else
    {
        if (steps_beyond_done == 0)
        {
            cout << "logger warn. step beyond done. reset should be called" << endl;
        }
        steps_beyond_done += 1;
        reward = 0.0;
    }

    return reward;
}

// Check if the simulation is done (problem specific)
bool CartpoleComponent::getDone(){
    bool done = false;

    if (steps>= 500 || state[0] < x_threshold * -1 || state[0] > x_threshold || state[2] < theta_threshold_radians * -1 || state[2] > theta_threshold_radians)
    {
        done = true;
    }

    return done;
}







// Utility functions and getters
void CartpoleComponent::resetStepVariables(){

}
ObsType CartpoleComponent::computeObservation(){
    return getRLState();

}
RewardType CartpoleComponent::computeReward(){
    return getReward();
}

void CartpoleComponent::finish(){

}

#endif