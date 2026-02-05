package JamesLib;
import inet.transportlayer.tcp.Tcp;

simple RLTcp extends Tcp
{
    parameters:
        @class(RLTcp);

        int maxWindow; 					// maximum window size
        int stateSize; 					// dimension of the state space
        int maxObsCount; 				// max number of observations for a single state
        double ewmaWeight; 				// weight for the moving average of state variables (as in Tcp-Ex Machina)
        double delayWeightReward; 		// weight for throughput vs. delay trade-off

        bool printDebugMessages;         // used to print debug messages
        int maxRLSteps;                 // how many training steps should take place before the agent reports as done

        @signal[stepData];				// step data signal
        @signal[actionQuery];			// action query signal
}