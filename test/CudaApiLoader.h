#pragma once

#include <memory>
#include <functional>
#include <stdexcept>

namespace gemu {

class DynamicLoader {
public:
    explicit DynamicLoader(const char* path);
    ~DynamicLoader();
    void* symbol(const char* name);
    template <typename Func>
    Func* function(const char* name) {
        auto s = this->symbol(name);
        if (!s)
            throw std::runtime_error(std::string("cannot load symbol: ") + name);
        return reinterpret_cast<Func*>(s);
    }
private:
    void* handle;
};

class CudaApiLoader
{
public:
    static void init(const char* path);
    static void cleanup();

    template <typename Fnc>
    static auto function(const char* name)
    {
       return _loader->function<Fnc>(name);
    }

private:
    static std::unique_ptr<DynamicLoader> _loader;
};

} // namespace gemu

// CUDA api

extern "C" int (*cuInit__zz)(unsigned);
