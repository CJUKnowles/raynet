package Orca;
import tcppaced.transportlayer.tcp.TcpPaced;

simple OrcaTcp extends TcpPaced
{
    parameters:
        @class(OrcaTcp);

        double delayCoefficient = default(1.25); 		    // Reward only suffers if (delay > minDelay * rewardDelayForgiveness). (higher values = more forgiveness, less impact of delay increases on reward)
        double lossCoefficient = default(5);                // How strongly loss impacts reward (higher = stronger impact)

        bool printDebugMessages = default(false);                            // used to print debug messages
        bool takeActions = default(true);                                   // Skips actions if set to false. Useful for comparing against the base case without altering code.
        int maxRLSteps = default(0);                                         // Deprecated compatibility parameter; Python owns episode step limits.
        double fixedIntervalDuration = default(.02);                          // Duration of every monitor interval in seconds.
        double delayWeight = default(5);                    // Beta term from Orca paper. Reward degrades when RTT > baseRTT*delayWeight. Larger values emphasize throughput in reward calculation.
        @signal[stepData];				                    // step data signal
        @signal[actionQuery];			                    // action query signal

        // Statistics recording
        @signal[throughput];
        @signal[intervalDuration];
        @statistic[throughput](record=vector(removeRepeats); interpolationmode=sample-hold);
        @statistic[intervalDuration](record=vector(removeRepeats); interpolationmode=sample-hold);
}
