// Written by Luca - Just a version of TCP that allows access to custom flavours (like my JamesCC)
#ifdef ASTREA
#ifndef TRANSPORTLAYER_AstreaTcp_H_
#define TRANSPORTLAYER_AstreaTcp_H_

#include <inet/transportlayer/tcp/Tcp.h>

using namespace inet::tcp;
using namespace omnetpp;
/*
 * Overrides TcpPaced implementation to define new NED parameters.
 */
class AstreaTcp : public Tcp
{
public:
    AstreaTcp();
    virtual ~AstreaTcp();
};

#endif /* TRANSPORTLAYER_AstreaTcp_H_ */
#endif
