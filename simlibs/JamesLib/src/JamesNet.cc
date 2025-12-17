#include "inet/common/InitStages.h"
#include "inet/networks/base/NetworkBase.h"

class JamesNet : public inet::NetworkBase
{
protected:
    virtual int numInitStages() const override {
        
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "This a number = " << 50 << "\n";
        EV_WARN << "INITSTAGE_LAST = " << inet::INITSTAGE_LAST << "\n";
        return 50;
    }

    virtual void initialize(int stage) override {
        EV_WARN << "=== NetworkBase initialize() ===\n";
        NetworkBase::initialize(stage);

        EV_WARN << "=== JamesNet initialize() ===\n";
        //EV_WARN << "CTX = " << simulation.getContextType() << "\n";
        EV_WARN << "stage = " << stage << "\n";
        EV_WARN << "INITSTAGE_LAST = " << inet::INITSTAGE_LAST << "\n";

        if (stage == inet::INITSTAGE_LAST) {
            EV_WARN << ">>> Reached INITSTAGE_LAST <<<\n";
        }
    }
};

Define_Module(JamesNet);