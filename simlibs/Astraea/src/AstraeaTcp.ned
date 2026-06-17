package Astraea;
import tcppaced.transportlayer.tcp.TcpPaced;

simple AstraeaTcp extends TcpPaced
{
    parameters:
        @class(AstraeaTcp);

        bool printDebugMessages = default(false);           // used to print debug messages
        bool takeActions = default(true);                   // Skips actions if set to false. Useful for comparing against the base case without altering code.
        int maxRLSteps = default(0);                        // Deprecated compatibility parameter; Python owns episode step limits.
        double actionControlCoeff = default(0.025);         // Original Astraea action alpha; raw action 1.0 increases cwnd by 2.5%.
        double fixedIntervalDuration = default(.03);        // Duration of every monitor interval in seconds.
        @signal[stepData];				                    // step data signal
        @signal[actionQuery];			                    // action query signal


        // Statistics recording
        @signal[throughput];
        @signal[intervalDuration];
        @statistic[throughput](record=vector(removeRepeats); interpolationmode=sample-hold);
        @statistic[intervalDuration](record=vector(removeRepeats); interpolationmode=sample-hold);
}
