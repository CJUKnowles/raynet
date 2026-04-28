package CleanSlate;
import tcppaced.transportlayer.tcp.TcpPaced;

simple CleanSlateTcp extends TcpPaced
{
    parameters:
        @class(CleanSlateTcp);

        int maxWindow; 					                    // maximum window size
        int stateSize; 					                    // dimension of the state space
        int maxObsCount; 				                    // max number of observations for a single state
        double ewmaWeight; 				                    // weight for the moving average of state variables (as in Tcp-Ex Machina)
        double rewardDelayForgiveness; 		                    // Reward only suffers if (delay > minDelay * rewardDelayForgiveness). (higher values = more forgiveness, less impact of delay increases on reward)
        double rewardLossMultiplier;                        // How strongly loss impacts reward (higher = stronger impact)

        bool printDebugMessages;                            // used to print debug messages
        bool takeActions;                                   // Skips actions if set to false. Useful for comparing against the base case without altering code.
        int maxRLSteps;                                     // how many training steps should take place before the agent reports as done
        bool fixedIntervals = default(true);         // If false, monitor intervals will adjust to SRTT dynamically
        double fixedIntervalDuration = default(.1);         // How many seconds a fixed monitor interval should last. If not fixed, MTPs will be set to SRTT dynamically
        double delayWeight = default(5);                    // Beta term from CleanSlate paper. Reward degrades when RTT > baseRTT*delayWeight. Larger values emphasize throughput in reward calculation.
        @signal[stepData];				                    // step data signal
        @signal[actionQuery];			                    // action query signal

        // Statistics recording
        @signal[throughput];
        @signal[intervalDuration];
        @statistic[throughput](record=vector(removeRepeats); interpolationmode=sample-hold);
        @statistic[intervalDuration](record=vector(removeRepeats); interpolationmode=sample-hold);
}