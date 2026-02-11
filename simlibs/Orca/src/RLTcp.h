// Written by Luca - Just a version of TCP that allows access to custom flavours (like my JamesCC)
#ifdef ORCA
#ifndef TRANSPORTLAYER_RLTCP_H_
#define TRANSPORTLAYER_RLTCP_H_

#include <inet/transportlayer/tcp/Tcp.h>
#include <inet/transportlayer/tcp/TcpConnection.h>

using namespace inet::tcp;
using namespace omnetpp;
/*
 * Overrides Tcp implementation to define new NED parameters.
 */
class RLTcp : public Tcp
{
public:
    RLTcp();
    virtual ~RLTcp();

    protected:
    /** Factory method; may be overriden for customizing Tcp */
    virtual TcpConnection *createConnection(int socketId);
};

#endif /* TRANSPORTLAYER_RLTCP_H_ */
#endif
