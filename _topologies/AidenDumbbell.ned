package topologies;

@namespace(inet);

import inet.node.inet.StandardHost;
import inet.node.inet.Router;
import ned.DatarateChannel;
import ned.IBidirectionalChannel;
import inet.common.scenario.ScenarioManager;
import tcppaced.networklayer.configurator.ipv4.AutoNetworkConfigurator;
import RLComponents.Broker;

network AidenDumbbell
{
    parameters:
        @display("bgb=512,395");
        int numberOfFlows = default(1);
    types:
        channel standardLine extends DatarateChannel //look at 
        {
            delay = 0ms;
            datarate = 10Gbps;
        }
        
        channel bottleneckLine extends DatarateChannel //look at 
        {
            delay = 0.5ms;
            datarate = 100Mbps;
        }
    submodules:
        client[numberOfFlows]: StandardHost {
            @display("p=68,71,m,n,numberOfFlows,150");
        }
        configurator: AutoNetworkConfigurator {
            @display("p=450,350");
        }
        server[numberOfFlows]: StandardHost {
            @display("p=431,50,m,n,numberOfFlows,150");
        }
        router1: Router {
            @display("p=215,140");
        }
        router2: Router {
            @display("p=327,140");
        }
        scenarioManager: ScenarioManager {
            @display("p=100,300;is=s");
        }
        broker: Broker;
    connections:
        for i=0..sizeof(client)-1 {
            client[i].pppg++ <--> standardLine <--> router1.pppg++;
        }

        for n=0..sizeof(server)-1 {
            server[n].pppg++ <--> standardLine <--> router2.pppg++;
        }
        
        router1.pppg++ <--> bottleneckLine <--> router2.pppg++;

}
