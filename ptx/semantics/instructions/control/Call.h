#ifndef PTXCALLINSTRH
#define PTXCALLINSTRH

#include "semantics/instructions/control/ControlInstruction.h"

namespace ptx {
    class Call : public ControlInstruction {
        PTX_DECLARE_DISPATCH
    public:
        Call (const std::string& target=std::string(),
              const std::string& result=std::string(),
              const std::vector<std::string>& parameters=std::vector<std::string>(),
              bool isDivergent=false)
            :_target(target)
            ,_result(result)
            ,_parameters(parameters)
            ,_isDivergent(isDivergent)
        {}
        std::string toString() const override {
            return "<call> " + this->target();
        }
        bool isDivergent() const { return this->_isDivergent; }
        std::string target() const { return this->_target; }
        std::string result() const { return this->_result; }
        bool hasResult() const { return this->_result.empty()==false; }
        size_t parameterCount() const { return this->_parameters.size(); }
        std::string parameter(int i) const { return this->_parameters[i]; }
    private:
        std::string _target;
        std::string _result;
        std::vector<std::string> _parameters;
        bool _isDivergent;
    };
}

#endif
