package Astrea;
import tcppaced.transportlayer.tcp.TcpPaced;

simple AstreaTcp extends TcpPaced
{
    parameters:
        @class(AstreaTcp);

        int maxWindow; 					                    // maximum window size
        int stateSize; 					                    // dimension of the state space
        int maxObsCount; 				                    // max number of observations for a single state
        double ewmaWeight; 				                    // weight for the moving average of state variables (as in Tcp-Ex Machina)
        double rewardDelayForgiveness; 		                    // Reward only suffers if (delay > minDelay * rewardDelayForgiveness). (higher values = more forgiveness, less impact of delay increases on reward)
        double rewardLossMultiplier;                        // How strongly loss impacts reward (higher = stronger impact)

        bool printDebugMessages;                            // used to print debug messages
        bool takeActions;                                   // Skips actions if set to false. Useful for comparing against the base case without altering code.
        int maxRLSteps;                                     // how many training steps should take place before the agent reports as done
        double monitorIntervalDuration = default(1.0);      // How many seconds a monitor interval should last
        @signal[stepData];				                    // step data signal
        @signal[actionQuery];			                    // action query signal


        @signal[throughput];
        @signal[srtt];
        @signal[action];
        @signal[intervalDuration];
        @signal[cwnd];
        @statistic[cwnd](record=vector(removeRepeats); interpolationmode=sample-hold);
        @statistic[throughput](record=vector(removeRepeats); interpolationmode=sample-hold);
        @statistic[srtt](record=vector(removeRepeats); interpolationmode=sample-hold);
        @statistic[action](record=vector(removeRepeats); interpolationmode=sample-hold);
        @statistic[intervalDuration](record=vector(removeRepeats); interpolationmode=sample-hold);
}