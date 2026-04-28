// Written by Luca - Just a version of TCP that allows access to custom flavours (like my JamesCC)
#ifdef CLEANSLATE
#ifndef TRANSPORTLAYER_CleanSlateTcp_H_
#define TRANSPORTLAYER_CleanSlateTcp_H_

#include <inet/transportlayer/tcp/Tcp.h>
#include <transportlayer/tcp/TcpPaced.h>

using namespace inet::tcp;
using namespace omnetpp;
/*
 * Overrides TcpPaced implementation to define new NED parameters.
 */
class CleanSlateTcp : public TcpPaced
{
public:
    CleanSlateTcp();
    virtual ~CleanSlateTcp();
};

#endif /* TRANSPORTLAYER_CleanSlateTcp_H_ */
#endif
