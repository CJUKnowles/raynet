package JamesLib;

import JamesLib.JamesNet;
import JamesLib.JamesCC;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networks.base.NetworkBase;
import inet.node.inet.StandardHost;
import inet.node.inet.Router;
import ned.IdealChannel;
import ned.DatarateChannel;
import ned.IBidirectionalChannel;
import inet.node.ethernet.Eth10M;

import RLComponents.Broker;
import RLComponents.Stepper;

network JamesNet
{
    parameters:
        // @statistic[packetDropped] (source = packetDropped; record = count);
        @display("bgb=800,395");
        int numberOfFlows = default(1);
    types:
        channel standardLine extends DatarateChannel
        {
            delay = 4.75ms;
            datarate = 10Gbps;
        }

        channel bottleneckLine extends DatarateChannel
        {
            delay = 0.5ms;
            datarate = 10Mbps;
        }
    submodules:
        configurator: Ipv4NetworkConfigurator {
            @display("p=0,0;is=s");
        }
        client[numberOfFlows]: StandardHost {
            @display("p=68,71,c,70,numberOfFlows,150");
        }
        server[numberOfFlows]: StandardHost {
            @display("p=431,71,c,70,numberOfFlows,150");
        }
        router1: Router {
            @display("p=200,150");
        }
        router2: Router {
            @display("p=300,150");
        }
        broker: Broker{};
		stepper: Stepper{};
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