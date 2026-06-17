package topologies;

@namespace(inet);

import inet.node.inet.StandardHost;
import inet.node.inet.Router;
import ned.DatarateChannel;
import ned.IBidirectionalChannel;
import inet.common.scenario.ScenarioManager;
import tcppaced.networklayer.configurator.ipv4.AutoNetworkConfigurator;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import RLComponents.Broker;

network Dumbbell
{
    parameters:
        int numberOfFlows = default(1);
        double bottleneckDatarate @unit("Mbps") = default(100Mbps);
        double bottleneckDelay @unit("ms") = default(.5ms);
        
    types:
        channel standardLine extends DatarateChannel
        {
            delay = 0ms;
            datarate = 10Gbps;
        }
        channel bottleneckLine extends DatarateChannel
        {
            delay = parent.bottleneckDelay; 
            datarate = parent.bottleneckDatarate;
        }
        
        
    submodules:
        // configurator: AutoNetworkConfigurator;
        configurator: Ipv4NetworkConfigurator;
        scenarioManager: ScenarioManager;
        
        client[numberOfFlows]: StandardHost;
        server[numberOfFlows]: StandardHost;
        router1: Router;
        router2: Router;
        broker: Broker;
    connections:
        // Bottleneck
        router1.pppg++ <--> bottleneck: bottleneckLine <--> router2.pppg++;
        
        // Left
        for i=0..sizeof(client)-1 {
            client[i].pppg++ <--> standardLine <--> router1.pppg++;
        }

        // Right
        for n=0..sizeof(server)-1 {
            server[n].pppg++ <--> standardLine <--> router2.pppg++ ;
        }
        

}