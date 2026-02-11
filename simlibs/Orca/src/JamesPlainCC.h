#ifdef ORCA
#ifndef __JAMES_PLAIN_CC_H_
#define __JAMES_PLAIN_CC_H_

#include <omnetpp.h>
#include <iostream>
#include <string>
#include <math.h>
#include <array>
#include <random>
#include <tuple>
#include "BrokerData.h"
#include "RLInterface.h"

#include "MonitorInterval.h"
#include "inet/transportlayer/tcp/flavours/TcpNewReno.h"
#include <inet/transportlayer/tcp/Tcp.h>
#include <inet/transportlayer/tcp/TcpConnection.h>

using namespace omnetpp;
using namespace inet::tcp;
using namespace inet;
using namespace learning;

class JamesPlainCC : public TcpNewReno
{
    // Just a class that extends any arbitrary TcpAlgo for easier access.
    // This is necessary because the usual algorithms are not visible to JamesTcpConn
};
#endif
#endif

