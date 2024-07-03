#include <iostream>
#include <cstdint>
#include <cstring>

#include "Ocl/Ocl.hpp"

#include <cstdint>
#include <vector>
#include <iostream>

#include <cstring>
#include <map>
#include <span>

#define MT(x) std::make_tuple(#x, Query<x>())

#include <chrono>
#include <thread>
#include <set>
template<auto>class Query{};

std::atomic<int32_t> syncCallback = 0;

template<auto Index>
void Callback(cl_program handle, Ocl::Kernel<Ocl::Handle<Ocl::Kernel<>>> *kernel){
     auto program = Ocl::Handle<Ocl::Program<>>(handle);
     auto &&devices = Ocl::Enum(program, Query<CL_PROGRAM_DEVICES>());
    for(auto &&device : devices){
        std::cout << "[Program]" << bool(program) << ": " << device << ": " << (Ocl::Handle<Ocl::Program<>>const&)program << /*": " << Ocl::Info(program) <<*/ ": "<< Ocl::Info(program, device, Query<CL_PROGRAM_BUILD_LOG>()) << "\n";
    }

    std::cout << "Program: CL_PROGRAM_NUM_KERNELS: " << Ocl::Info(program, Query<CL_PROGRAM_NUM_KERNELS>()) << "\n";
    std::cout << "CL_PROGRAM_SOURCE: " << Ocl::Info(program, Query<CL_PROGRAM_SOURCE>()) << "\n";
    std::cout << "CL_PROGRAM_KERNEL_NAMES: " << Ocl::Info(program, Query<CL_PROGRAM_KERNEL_NAMES>()) << "\n";

    *kernel = Ocl::Kernel(program, Ocl::Enum(program, Query<CL_PROGRAM_KERNEL_NAMES>())[0]);
    std::cout << "[Kernel]" << bool(*kernel) << ": " << (Ocl::Handle<Ocl::Kernel<>>const&)*kernel << ": " << Ocl::Info(*kernel) << "\n";

    for(auto &&device : devices)
    {
        std::cout << "CL_KERNEL_WORK_GROUP_SIZE: " << Ocl::Info(*kernel, device, Query<CL_KERNEL_WORK_GROUP_SIZE>()) << "\n";
        std::cout << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << Ocl::Info(*kernel, device, Query<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>()) << "\n";
        std::cout << "CL_KERNEL_LOCAL_MEM_SIZE: " << Ocl::Info(*kernel, device, Query<CL_KERNEL_LOCAL_MEM_SIZE>()) << "\n";
        std::cout << "CL_KERNEL_PRIVATE_MEM_SIZE: " << Ocl::Info(*kernel, device, Query<CL_KERNEL_PRIVATE_MEM_SIZE>()) << "\n";
    }
    syncCallback.fetch_add(1, std::memory_order::relaxed);
    syncCallback.notify_one();
}

namespace Ocl3{
int32_t RadixSort(){
    auto i = 0;
    for(auto &&platform : Ocl::Enum(Ocl::Platform())){
        std::cout << "Platform[" << i++ << "]: " << platform << "\n";
        std::cout << "\tName:           " << Ocl::Info(platform, Query<CL_PLATFORM_NAME>()) << "\n";
        std::cout << "\tVendor:         " << Ocl::Info(platform, Query<CL_PLATFORM_VENDOR>()) << "\n";
        std::cout << "\tDriver Version: " << Ocl::Info(platform, Query<CL_PLATFORM_VERSION>()) << "\n";
      //  if(i==1) continue;
        auto devices = Ocl::Enum(Ocl::Device(), platform);
        for(auto &&device : devices){
            std::cout << "\tDevice: " << device << "\n";
            constexpr auto queries = std::make_tuple(MT(CL_DEVICE_NAME), MT(CL_DEVICE_VENDOR), MT(CL_DEVICE_PROFILE),
                                                     MT(CL_DEVICE_VERSION), MT(CL_DRIVER_VERSION), MT(CL_DEVICE_VENDOR_ID),
                                                     MT(CL_DEVICE_TYPE), MT(CL_DEVICE_PARTITION_MAX_SUB_DEVICES), MT(CL_DEVICE_MAX_ON_DEVICE_QUEUES), MT(CL_DEVICE_MAX_COMPUTE_UNITS), MT(CL_DEVICE_MAX_CLOCK_FREQUENCY), MT(CL_DEVICE_ADDRESS_BITS), MT(CL_DEVICE_MAX_NUM_SUB_GROUPS), MT(CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES), MT(CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES),
                                                     MT(CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), MT(CL_DEVICE_MAX_WORK_GROUP_SIZE), MT(CL_DEVICE_MAX_MEM_ALLOC_SIZE), MT(CL_DEVICE_GLOBAL_MEM_SIZE), MT(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE), MT(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE), MT(CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE), MT(CL_DEVICE_LOCAL_MEM_SIZE),
                                                     MT(CL_DEVICE_SVM_CAPABILITIES), MT(CL_DEVICE_OPENCL_C_VERSION));
            std::apply([&device](auto&&... query) {((std::cout << "\t\t" << std::get<0>(query)<< ": " << Ocl::Info(device, std::get<1>(query)) << '\n'), ...);}, queries);
        
            
            for(auto &&size : Ocl::Enum(device, Query<CL_DEVICE_MAX_WORK_ITEM_SIZES>())){
                std::cout << size << ",";
            }
            std::cout << "\n";

            for(auto &&entry : Ocl::Enum(device, Query<CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION>())){
                std::cout << entry.name << ": " << entry.version << ",";
            }
            std::cout << "\n";
        }
        auto sourcesById =
            std::map<std::string, std::string>{{"Count", "/*uint constant masks[4] = {0xFF, 0xFF00, 0xFF0000, 0xFF000000};*/"                        
                "kernel void Count(global uint *const input, global uint *const input2, global uchar *const result, global uint *const counts, int const blockSize, int const pass) { "
                "global uint *const _input = (*result & 2) ? input2 : input;"
                "const size_t globalOffsetTgt = get_global_id(0) * get_global_size(0);"
                "const size_t globalOffsetSrc = get_global_id(0) * blockSize;"
                "/*const size_t localOffset = get_local_id(0) << 8;*/"
                "private uint _counts[256];"
                "for(int j=0; j<256; ++j) _counts[j] = 0;/**/"
                "/*const uint shift = pass << 3;"
                "private uint const masks[4] = {0xFF, 0xFF00, 0xFF0000, 0xFF000000};*/"
                "for(int j=0; j<blockSize; ++j){"
                    "/*uint const value = _input[globalOffsetSrc + j];"
                "uint const symbol = (value & masks[pass]) >> shift;"
                "++_counts[symbol];*/"
                " ++_counts[(((global uchar *)&_input[globalOffsetSrc + j])[pass])];"
                "/* ++counts[globalOffsetTgt + (((global uchar *)&_input[globalOffsetSrc + j])[pass])];*/"
                "}"
                "for(int j=0; j<256; ++j){"
                "   counts[globalOffsetTgt + j] = _counts[j];/**/"
                    "/*offsets[j] += _counts[j]*/"
                "}"
            "}"},
           {"ExclusiveScan", "kernel void ExclusiveScan(global uint *const counts, global uint *const offsets, global uchar *const result, uint const blockCount){"
           "uint const symbol = get_global_id(0);"
           "uint old = 0; uint current = 0; uint base = 0; int z = 0;"
            "for(uint index = 0; index < (1u<<8); ++index){"
                "old = current;"
                "current = old + counts[(index << 8) + symbol];"
                "offsets[(index << 8) + symbol] = old;"
            "}"
            "if (symbol == 0) *result = (*result & 2 ? 1 : 2);"
                "/*if(base == current) ++z;"
                "else base = current;*/"
           "/**result = (z != 255) ? (*result & 2 ? 1 : 3) : (*result & 2 ? 2 : 0);*/"
        "}"
       },
              {"Sort", "kernel void Sort(global uint *const keysOut, global uint *const counts, global uint *const offsets, global uint *const keysIn, global uchar *const result, uint const blockSize, uint const index){"
            "/*if ((*result & 1) == 0) return;*/"
            "global uint *const _output = (*result & 2) ? keysIn : keysOut;"
            "global uint *const _input = (*result & 2) ? keysOut : keysIn;"
            "const uint offset = get_global_id(0) * 256;"
            "private uint _offsets[256];"
            "private uint _counts[256];"
            "{"
            "_offsets[0] = 0;"
            "_counts[0] = 0;"
            "uint old = 0; uint current = counts[offset];"
            "for(int symbol = 1; symbol < 256; ++symbol){"
                "old = current;"
                "current = old + counts[offset + symbol];"
                "_offsets[symbol] = old;"
                "_counts[symbol] = _counts[symbol - 1] + counts[((get_global_size(0)-1)<<8)+symbol-1] + offsets[((get_global_size(0)-1)<<8)+symbol-1];"
            "}"
            "}"
            "const uint offset2 = get_global_id(0) * blockSize;"
            "local uint out[1<<12];"
            "for(int i=0; i<blockSize; ++i){"
                "/*uchar const symbol = ((global uchar*)&_input[offset2 + i])[index];*/"
                "uint const value = _input[offset2 + i];"
                "uint const symbol = (value & (0xFF << (index * 8))) >> (index * 8);"
                "out[_offsets[symbol]++] =value/*_input[offset2 + i]*/;"
            "}"
            "uint old = 0;"
            "for(int symbol=0; symbol<256; ++symbol){"
                "const uint o = offsets[offset + symbol] + _counts[symbol];"
                "for(int j=0; j<_offsets[symbol] - old; ++j){"
                    "_output[o + j] = out[old + j];"
                "}"
                "old = _offsets[symbol];"
            "}"
        "}"}
                        };

         {
            auto context = Ocl::Context(devices);
            auto kernels = std::array<Ocl::Kernel<Ocl::Handle<Ocl::Kernel<>>>, 3>();

            constexpr auto compilerOptions = std::string_view("-cl-std=CL2.0 -cl-mad-enable");
            auto programs = std::tuple{ context(sourcesById["Count"])(Callback<0>)(compilerOptions, kernels[0]),
                                        context(sourcesById["ExclusiveScan"])(Callback<2>)(compilerOptions, kernels[1]),
                                        context(sourcesById["Sort"])(Callback<1>)(compilerOptions,  kernels[2])};
            
            while(true){
                auto res = syncCallback.load(std::memory_order::acquire);
                if(res == 3) break;
                syncCallback.wait(res);
            }
            syncCallback.store(0, std::memory_order::relaxed);

            const uint32_t count = ((1 << 20)); // <= blockCount * blockSize
            const uint32_t blockSize = 1 << 12; //blockSize
            const uint32_t blockCount = count / blockSize;  //blockCount
            const uint32_t workGroupSize = 1;/**/

            auto deviceInput = Ocl::Memory(context, Ocl::SizeOf<cl_uint>(count), Query<CL_MEM_READ_WRITE>());
            auto deviceTarget = Ocl::Memory(context, Ocl::SizeOf<cl_uint>(count), Query<CL_MEM_READ_WRITE>());
            auto deviceCounts = Ocl::Memory(context, Ocl::SizeOf<cl_uint>(blockCount * 256), Query<CL_MEM_READ_WRITE>());
            auto deviceOffsets = Ocl::Memory(context, Ocl::SizeOf<cl_uint>(blockCount * 256), Query<CL_MEM_READ_WRITE>());
            auto deviceResult = Ocl::Memory(context, Ocl::SizeOf<cl_uchar>(1), Query<CL_MEM_READ_WRITE>());
            std::cout << "[Buffer]" << bool(deviceCounts) << ": " << (Ocl::Handle<Ocl::Memory<>>const&)deviceCounts << ": " << Ocl::Info(deviceCounts) << "\n";

            auto j = 0;
            for(auto &&device : devices){
              //if(++j==2 || i==1)continue;
{
                auto commandQueue = context(device);
               // std::cout << "[CommandQueue]" << bool(commandQueue) << device << ": " << (Ocl::Handle<Ocl::CommandQueue<>>const&)commandQueue << ": " << Ocl::Info(commandQueue) << "\n";      
                             
                uint32_t *result = nullptr;
                uint32_t *input = nullptr;
                cl_uint *counts = nullptr;
                cl_uchar *resultExcl = nullptr;
                commandQueue(deviceResult)(resultExcl);
                commandQueue(deviceInput)(input);

                kernels[1](deviceCounts, deviceOffsets, deviceResult, blockCount);
                Ocl::Wait{commandQueue};

                for(auto i=0; i<count; ++i)
                    input[i] = count - i;

                *resultExcl = 0;

                commandQueue(deviceInput)(std::move(input));
                commandQueue(deviceResult)(std::move(resultExcl));
                kernels[0](deviceInput, deviceTarget, deviceResult, deviceCounts, blockSize);
                kernels[2](deviceInput, deviceCounts, deviceOffsets, deviceTarget, deviceResult, blockSize);
                Ocl::Wait{commandQueue};
                
      auto &&begin = std::chrono::steady_clock::now();
      {
        for(auto pass = 0; pass < 4; ++pass){
            kernels[0](std::make_tuple(5, pass));
            commandQueue(kernels[0])(Ocl::Size(blockCount), Ocl::Size(1u << 5));
            commandQueue(kernels[1])(Ocl::Size(1u << 8), Ocl::Size(1u<<5));
            kernels[2](std::make_tuple(6, pass));
            auto event1 = commandQueue(kernels[2])/*(event1)*/(Ocl::Size(blockCount), Ocl::Size(1u));
            Ocl::Wait{event1};
        }
            auto event1 = commandQueue(deviceResult)(resultExcl);
            event1 = commandQueue(*resultExcl & 2 ? deviceTarget : deviceInput)(event1)(result);
            Ocl::Wait{event1};
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "\n";
        for(auto i=0; i<10; ++i) std::cout << result[i] << ",";
        std::cout << "\n";
        commandQueue(*resultExcl & 2 ? deviceTarget : deviceInput)(std::move(result));
}           
             }
        }
     }

    std::cout << "Done.\n";
    return 0;
}

inline auto result = RadixSort(); 
}

int32_t main(){}