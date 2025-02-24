#include <stdio.h>
#include <dlfcn.h>
#include <nccl.h>

// Function pointer for the real ncclBroadcast
typedef ncclResult_t (*ncclBroadcast_t)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
static ncclBroadcast_t real_ncclBroadcast = nullptr;

// Ensure libnccl.so.2 is loaded
static void load_real_nccl_functions() {
    if (!real_ncclBroadcast) {
        void* handle = dlopen("libnccl.so.2", RTLD_LAZY);
        if (!handle) {
            printf("[HOOK] Failed to load libnccl.so.2: %s\n", dlerror());
            return;
        }
        real_ncclBroadcast = (ncclBroadcast_t)dlsym(handle, "ncclBroadcast");
        if (!real_ncclBroadcast) {
            printf("[HOOK] Failed to find original ncclBroadcast in libnccl.so.2\n");
        }
    }
}

// Hooked function
extern "C" ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
    load_real_nccl_functions();

    if (!real_ncclBroadcast) {
        printf("[HOOK] ncclBroadcast: Original function not found, returning error.\n");
        return ncclSystemError;
    }

    printf("[HOOK] Intercepted ncclBroadcast: count=%zu, root=%d\n", count, root);

    // Call the original function
    return real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}
