package Astraea;
import tcppaced.transportlayer.tcp.TcpPaced;

simple AstraeaTcp extends TcpPaced
{
    parameters:
        @class(AstraeaTcp);

        int maxWindow = default(0); 					        // Deprecated compatibility parameter.
        int stateSize = default(0); 					        // Deprecated compatibility parameter.
        int maxObsCount = default(0); 				        // Deprecated compatibility parameter.
        double ewmaWeight = default(0); 				    // Deprecated compatibility parameter.
        double delayCoefficient = default(1.25);            // Deprecated compatibility parameter; Python owns rewards.
        double lossCoefficient = default(5);                // Deprecated compatibility parameter; Python owns rewards.
        bool printDebugMessages = default(false);           // used to print debug messages
        bool takeActions = default(true);                   // Skips actions if set to false. Useful for comparing against the base case without altering code.
        int maxRLSteps = default(0);                        // Deprecated compatibility parameter; Python owns episode step limits.
        double actionControlCoeff = default(0.025);         // Original Astraea action alpha; raw action 1.0 increases cwnd by 2.5%.
        double monitorIntervalDuration = default(1.0);      // Deprecated compatibility parameter; use fixedIntervalDuration.
        double fixedIntervalDuration = default(.03);        // Duration of every monitor interval in seconds.
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
