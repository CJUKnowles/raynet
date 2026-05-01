package topologies;

@namespace(inet);

import inet.common.misc.ThruputMeteringChannel;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import RLComponents.Broker;
import CleanSlate.CleanSlateNet;
import CleanSlate.CleanSlate;
import CleanSlate.CleanSlateTcp;

// NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!
// THIS IS JUST ANOTHER DUMBBELL! THIS IS A PLACEHOLDER FOR EXAMPLE!
// PUT YOUR OWN GENERALLY USEFUL TOPOLOGY HERE! AN EXPERIMENT/.INI WILL USE THIS
// AND OVERRIDE CONFIGURABLE VALUES!
network Manhattan
{
    parameters:
        int numberOfFlows = default(1);
        double bottleneckDatarate @unit("Mbps") = default(3Mbps);
        double bottleneckDelay @unit("ms") = default(10ms);
        
    types:
        channel bottleneckLine extends ThruputMeteringChannel
        {
            delay = parent.bottleneckDelay; 
            datarate = parent.bottleneckDatarate; 
        }
        
        channel standardLine extends ThruputMeteringChannel
        {
            delay = 0ms;
            datarate = 1000Mbps;
        }
    submodules:
    
        configurator: Ipv4NetworkConfigurator;
        
        client[numberOfFlows]: StandardHost;
        server[numberOfFlows]: StandardHost;
        router1: Router;
        router2: Router;
        broker: Broker;
    connections:
        // Left
        for i=0..sizeof(client)-1 {
            client[i].pppg++ <--> standardLine <--> router1.pppg++;
        }

        // Right
        for n=0..sizeof(server)-1 {
            router2.pppg++ <--> standardLine <--> server[n].pppg++;
        }
        
        // Bottleneck
        router1.pppg++ <--> bottleneckLine <--> router2.pppg++;
}