package Orca;
import tcppaced.transportlayer.tcp.TcpPaced;

simple OrcaTcp extends TcpPaced
{
    parameters:
        @class(OrcaTcp);

        int maxWindow; 					                    // maximum window size
        int stateSize; 					                    // dimension of the state space
        int maxObsCount; 				                    // max number of observations for a single state
        double ewmaWeight; 				                    // weight for the moving average of state variables (as in Tcp-Ex Machina)
        double delayWeightReward; 		                    // weight for throughput vs. delay trade-off

        bool printDebugMessages;                            // used to print debug messages
        int maxRLSteps;                                     // how many training steps should take place before the agent reports as done
        double monitorIntervalDuration = default(1.0);      // How many seconds a monitor interval should last
        double delayWeight = default(1.5);                  // Beta term from Orca paper. Reward degrades when RTT > baseRTT*delayWeight. Larger values emphasize throughput in reward calculation.
        @signal[stepData];				                    // step data signal
        @signal[actionQuery];			                    // action query signal
}