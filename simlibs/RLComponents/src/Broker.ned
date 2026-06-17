package RLComponents;

simple Broker
{
    @signal[brokerToStepper];
    @signal[simsecPerSec];
    @statistic[simsecPerSec](record=vector(removeRepeats); interpolationmode=sample-hold);

    string obsCollectionMode @enum("IMMEDIATE", "GROUPED", "INTERVALED") = default("IMMEDIATE");
    double simThroughputProbeInterval @unit(s) = default(0s); // Disabled when 0s.
}
