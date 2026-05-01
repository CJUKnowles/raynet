#include "TcpCubic.h"
namespace inet {
namespace tcp {
    class TcpPacedNoCC : public TcpCubic
    {
        protected:
            /**
            * A version of TcpPaced that does perform any congestion control (does not alter its cwnd automatically).
            * This was made to enable the easy creation of paced clean-slate RL protocols, like Astrea.
            * This should be identical to TcpCubic, but with any lines relating to changing the cwnd commented out.
            */

            /** Removes any cwnd alterations, otherwise unchanged */
            virtual void processRexmitTimer(TcpEventCode &event) override;
            virtual void receivedDataAck(uint32_t firstSeqAcked) override;
            virtual void receivedDuplicateAck() override;
    };
} // inet
} // tcp