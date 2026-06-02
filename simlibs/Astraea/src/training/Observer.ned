package Astraea.training;

simple Observer
{
    // Ground-truth network parameters, set by the config.ini. Used for computing rewards.
    double AstraeaBottleneckDatarate @unit("Mbps") = default(3Mbps);
    double AstraeaBottleneckDelay @unit("ms") = default(10ms);
    double AstraeaDataCapacity @unit("b") = default(10b);

    // Misc
    bool printDebugMessages;                            // used to print debug messages
}
