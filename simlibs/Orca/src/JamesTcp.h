#ifdef ORCA
#ifndef TRANSPORTLAYER_JAMETCP_H_
#define TRANSPORTLAYER_JAMETCP_H_

#include <inet/transportlayer/tcp/Tcp.h>
#include <inet/transportlayer/tcp/TcpConnection.h>

using namespace inet::tcp;
using namespace omnetpp;

class JamesTcp : public Tcp {
public:
    JamesTcp();
    virtual ~JamesTcp();

protected:
    /** Override factory method to create custom connections */
    virtual TcpConnection* createConnection(int socketId);
};

#endif // TRANSPORTLAYER_RLTCP_H_
#endif