package RLComponents;

simple Broker
{
    @signal[brokerToStepper];
    @signal[simsecPerSec];
    @statistic[simsecPerSec](record=vector(removeRepeats); interpolationmode=sample-hold);

    string obsCollectionMode @enum("INDEPENDENT", "GROUPED", "INTERVAL") = default("INTERVAL");
    double simThroughputProbeInterval @unit(s) = default(0s); // Disabled when 0s.
}
