package JamesLib;

@namespace(inet);
@namespace(ndp);

import JamesLib.JamesNet;
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
//import cartpole.CartpoleComponent; // Used as a test, delete once things are working
network JamesNet
{
    parameters:
        @statistic[packetDropped] (source = packetDropped; record = count);
        @display("bgb=800,395");
        //@class(JamesNet);
        int numberOfFlows = default(1);
    types:
        channel standardLine extends DatarateChannel //look at
        {
            delay = 4.75ms;
            datarate = 10Gbps;
        }

        channel bottleneckLine extends DatarateChannel //look at
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
////        cartpole: CartpoleComponent; // test, delete later
////
		stepper: Stepper{};
    connections:
        //pppg = point-to-point protocol. Alternative to ethernet (ethg+++)
        // Left

//        for i=0..sizeof(client)-1 {
//            client[i].ethg++ <--> Eth10M <--> router1.ethg++;
//        }
//
//        for n=0..sizeof(server)-1 {
//            router2.ethg++ <--> Eth10M <--> server[n].ethg++;
//        }
//
//        router1.ethg++ <--> Eth10M <--> router2.ethg++;
        for i=0..sizeof(client)-1 {
            client[i].pppg++ <--> standardLine <--> router1.pppg++;
        }

        for n=0..sizeof(server)-1 {
            router2.pppg++ <--> standardLine <--> server[n].pppg++;
        }

        router1.pppg++ <--> bottleneckLine <--> router2.pppg++;
}