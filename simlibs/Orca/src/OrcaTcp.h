// Written by Luca - Just a version of TCP that allows access to custom flavours (like my JamesCC)
#ifdef ORCA
#ifndef TRANSPORTLAYER_OrcaTcp_H_
#define TRANSPORTLAYER_OrcaTcp_H_

#include <inet/transportlayer/tcp/Tcp.h>
#include <transportlayer/tcp/TcpPaced.h>

using namespace inet::tcp;
using namespace omnetpp;
/*
 * Overrides TcpPaced implementation to define new NED parameters.
 */
class OrcaTcp : public TcpPaced
{
public:
    OrcaTcp();
    virtual ~OrcaTcp();
};

#endif /* TRANSPORTLAYER_OrcaTcp_H_ */
#endif
