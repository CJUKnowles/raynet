
#include "RLTcp.h"

Define_Module (RLTcp);

RLTcp::RLTcp()
{
} 
//test 

RLTcp::~RLTcp()
{
}

TcpConnection *RLTcp::createConnection(int socketId)
{
    auto moduleType = cModuleType::get("TcpPaced.TcpConnectionResultsRecording");
    char submoduleName[24];
    sprintf(submoduleName, "conn-%d", socketId);
    auto module = check_and_cast<TcpConnection *>(moduleType->createScheduleInit(submoduleName, this));
    module->initConnection(this, socketId);
    return module;
}
