#ifndef CUDAEVENT_H
#define CUDAEVENT_H

#include <time.h>
#include "cudaDefines.h"

namespace gemu{
namespace cuda{
    class Event {
    public:
        Event(const unsigned flags) : _flags(flags) {}
        void record() {
            this->_recordTime = clock();
            this->_wasRecorded = true;
        }
        float msTo(const Event& other) const {
            return (1000.0f*(other._recordTime - this->_recordTime)) / CLOCKS_PER_SEC;
        }
        void setStream(CUstream stream) {
            this->_stream = stream;
        }
        CUstream streamId() const {
            return this->_stream;
        }
        bool wasRecorded() const {
            return this->_wasRecorded;
        }
    private:
        const unsigned _flags;
        clock_t _recordTime;
        bool _wasRecorded = false;
        CUstream _stream = (CUstream)-1;
    };
}
}

#endif // CUDAEVENT_H

