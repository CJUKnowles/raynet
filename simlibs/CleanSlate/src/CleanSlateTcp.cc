#ifdef CLEANSLATE
#include "CleanSlateTcp.h"

Define_Module (CleanSlateTcp);

CleanSlateTcp::CleanSlateTcp(){} 
CleanSlateTcp::~CleanSlateTcp(){}

void TcpPacedConnection::computeRetransmissionRate()
{
    const double dt = simTime().dbl() - lastRetransmissionRateTime.dbl();
    if (dt <= 0)
        return;

    const uint64_t totalRtxBytes = totalRetransmittedBytesCounter;
    const uint64_t deltaRtxBytes = totalRtxBytes - lastTotalRetransmittedBytes;

    currRetransmissionRate = (double)deltaRtxBytes * 8.0 / dt; // bits/s
    emit(retransmissionRateSignal, currRetransmissionRate);

    prevLastTotalRetransmittedBytes = lastTotalRetransmittedBytes;
    lastTotalRetransmittedBytes = totalRtxBytes;
    lastRetransmissionRateTime = simTime();
}

#endif