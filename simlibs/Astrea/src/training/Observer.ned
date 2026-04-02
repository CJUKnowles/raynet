package Astrea.training;

simple Observer
{
    // Ground-truth network parameters, set by the config.ini. Used for computing rewards.
    double astreaBottleneckDatarate @unit("Mbps") = default(3Mbps);
    double astreaBottleneckDelay @unit("ms") = default(10ms);
    double astreaDataCapacity @unit("b") = default(10b);

    // Misc
    bool printDebugMessages;                            // used to print debug messages
}
