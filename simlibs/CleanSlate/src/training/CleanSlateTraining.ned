package CleanSlate.training;

@namespace(inet);

import inet.common.scenario.ScenarioManager;
import inet.common.misc.ThruputMeteringChannel;
import ned.DatarateChannel;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import RLComponents.Broker;

network CleanSlateTraining
{
    parameters:
        int numberOfFlows = default(1);
        double bottleneckDatarate @unit("Mbps") = default(3Mbps);
        double bottleneckDelay @unit("ms") = default(10ms);
        
    types:
        channel bottleneckLine extends DatarateChannel
        {
            delay = parent.bottleneckDelay; 
            datarate = parent.bottleneckDatarate; 
        }
        
        channel standardLine extends DatarateChannel
        {
            delay = 0ms;
            datarate = 1000Gbps;
        }
    submodules:
    
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
            router2.pppg++ <--> standardLine <--> server[n].pppg++;
        }
        

}