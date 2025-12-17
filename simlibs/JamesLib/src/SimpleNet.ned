
// Very simple network (tictoc) without INET components
package JamesLib;

import JamesLib.Txc;
import RLComponents.Broker;
import RLComponents.Stepper;

network SimpleNet
{
    submodules:
        broker: Broker;
        stepper: Stepper;
        tic: Txc;
        toc: Txc {
            @display("p=98,37");
        }
    connections:
        tic.out --> {  delay = 100ms; } --> toc.in;
        tic.in <-- {  delay = 100ms; } <-- toc.out;
}
