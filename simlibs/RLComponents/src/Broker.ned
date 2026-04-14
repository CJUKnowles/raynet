package RLComponents;

simple Broker
{
    @signal[brokerToStepper];

    string obsCollectionMode @enum("IMMEDIATE", "GROUPED", "INTERVALED") = default("IMMEDIATE");
}
