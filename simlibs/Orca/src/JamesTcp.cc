#ifdef ORCA
#include "JamesTcp.h"

Define_Module(JamesTcp);

JamesTcp::JamesTcp() {}
JamesTcp::~JamesTcp() {}

TcpConnection* JamesTcp::createConnection(int socketId) {
    auto moduleType = cModuleType::get("Orca.JamesTcpConn");
    //auto moduleType = cModuleType::get("inet.transportlayer.tcp.TcpConnection");

    char submoduleName[24];
    sprintf(submoduleName, "conn-%d", socketId);
    auto conn = check_and_cast<TcpConnection*>(moduleType->createScheduleInit(submoduleName, this));
    conn->initConnection(this, socketId);
    return conn;
}
#endif