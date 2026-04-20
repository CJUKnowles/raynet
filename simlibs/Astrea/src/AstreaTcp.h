// Written by Luca - Just a version of TCP that allows access to custom flavours (like my JamesCC)
#ifdef ASTREA
#ifndef TRANSPORTLAYER_AstreaTcp_H_
#define TRANSPORTLAYER_AstreaTcp_H_

#include <inet/transportlayer/tcp/Tcp.h>
#include <transportlayer/tcp/TcpPaced.h>

using namespace inet::tcp;
using namespace omnetpp;
/*
 * Overrides TcpPaced implementation to define new NED parameters.
 */
class AstreaTcp : public TcpPaced
{
public:
    AstreaTcp();
    virtual ~AstreaTcp();
};

#endif /* TRANSPORTLAYER_AstreaTcp_H_ */
#endif
