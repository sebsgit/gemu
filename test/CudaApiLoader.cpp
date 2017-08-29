#include "CudaApiLoader.h"

#ifdef WIN32
#include <windows.h>
#else
#error "Implement loader for other platforms"
#endif

using namespace gemu;

DynamicLoader::DynamicLoader(const char *path)
    :handle(nullptr)
{
#ifdef WIN32
    handle = LoadLibraryA(path);
    if (!handle)
        throw std::runtime_error(std::string("cannot load: ") + path);
#endif
}

DynamicLoader::~DynamicLoader()
{
#ifdef WIN32
    if (this->handle)
        FreeLibrary(static_cast<HMODULE>(this->handle));
#endif
}

void* DynamicLoader::symbol(const char *name)
{
#ifdef WIN32
    return GetProcAddress(static_cast<HMODULE>(this->handle), name);
#endif
}

std::unique_ptr<DynamicLoader> CudaApiLoader::_loader;

//TODO call convention
int (*cuInit__zz)(unsigned) = nullptr;

void CudaApiLoader::init(const char *path){
    _loader.reset(new DynamicLoader(path));
    cuInit__zz = _loader->function<int(unsigned)>("cuInit");
}

void CudaApiLoader::cleanup()
{
    _loader.reset();
}
