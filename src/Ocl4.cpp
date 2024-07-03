#include <iostream>
#include <cstdint>
#include <cstring>

#include "Ocl.hpp"

#include <cstdint>
#include <vector>
#include <iostream>

#include <cstring>
#include <map>
#include <span>

#include <Hatch/Interface/Interface.hpp>
#define HATCH_OS_WINDOWS
#include <Hatch/Os/Os.hpp>

#include "Factory/Factory.hpp"
using Module = Interface::Module<Os::Windows<Os::User<>>>;
#include <Hatch/Logic/Logic.hpp>
#define __HATCH_TYPE__ 2
#include <Hatch/Type/Type.hpp>
#include <Hatch/Sync/Sync.hpp>
#include "Concurrency/Concurrency.hpp"
#include <Hatch/Metrics/Metrics.hpp>

#define MT(x) std::make_tuple(#x, Query<x>())


#include <chrono>
#include <thread>
template<auto>class Query{};

namespace Ocl4{

    bool ExclusiveScan(auto &&counts, uint32_t blockCount){
        uint64_t old = 0; uint64_t current = 0; uint8_t z = 0;
        for(auto symbol = 0; symbol < 256; ++symbol){
            for(auto block=0; block<blockCount; ++block){
                old = current;
                current = old + counts[symbol + block*256];
                counts[symbol + block*256] = old;
            }
            if(counts[1] == current) ++z;
        }
        return z != 255;
    }

    bool ExclusiveScan(auto &&counts, auto &&offsets, uint32_t blockCount){
    uint64_t old = 0; uint64_t current = 0; uint8_t z = 0;
    for(auto symbol = 0; symbol < 256; ++symbol){
        for(auto block=0; block<blockCount; ++block){
            old = current;
            auto const o = symbol + block*256;
            current = old + counts[o];
            offsets[o] = old;
        }
        if(offsets[1] == current) ++z;
    }
    return z != 255;
}


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
            constexpr auto queries = std::make_tuple(std::make_tuple("CL_DEVICE_NAME", Query<CL_DEVICE_NAME>()), std::make_tuple("CL_DEVICE_VENDOR", Query<CL_DEVICE_VENDOR>()), std::make_tuple("CL_DEVICE_PROFILE", Query<CL_DEVICE_PROFILE>()),
                                                     std::make_tuple("CL_DEVICE_VERSION", Query<CL_DEVICE_VERSION>()), std::make_tuple("CL_DRIVER_VERSION", Query<CL_DRIVER_VERSION>()), std::make_tuple("CL_DEVICE_VENDOR_ID", Query<CL_DEVICE_VENDOR_ID>()),
                                                     MT(CL_DEVICE_TYPE), MT(CL_DEVICE_PARTITION_MAX_SUB_DEVICES), MT(CL_DEVICE_MAX_ON_DEVICE_QUEUES), MT(CL_DEVICE_MAX_COMPUTE_UNITS), MT(CL_DEVICE_MAX_CLOCK_FREQUENCY), MT(CL_DEVICE_ADDRESS_BITS), MT(CL_DEVICE_MAX_NUM_SUB_GROUPS), MT(CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES), MT(CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES),
                                                     MT(CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), MT(CL_DEVICE_MAX_WORK_GROUP_SIZE), MT(CL_DEVICE_MAX_MEM_ALLOC_SIZE), MT(CL_DEVICE_GLOBAL_MEM_SIZE), MT(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE), MT(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE), MT(CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE), MT(CL_DEVICE_LOCAL_MEM_SIZE),
                                                     std::make_tuple("CL_DEVICE_SVM_CAPABILITIES", Query<CL_DEVICE_SVM_CAPABILITIES>()), MT(CL_DEVICE_OPENCL_C_VERSION));
            std::apply([&device](auto&&... query) {((std::cout << "\t\t" << std::get<0>(query)<< ": " << Ocl::Info(device, std::get<1>(query)) << '\n'), ...);}, queries);
        
            
            for(auto &&size : Ocl::Enum(device, Query<CL_DEVICE_MAX_WORK_ITEM_SIZES>())){
                std::cout << size << ",";
            }
            std::cout << "\n";
        }
        auto sourcesById2 =
                        std::map<std::string, std::string>{{"Count", "kernel void Count(global uint *const input, global uint *const counts, int const workGroupSize, int const pass) { "\
                         "/*local uint _counts[256];*/"
                        "const uint offset = get_global_id(0) * 256;" \
                        "const uint offset2 = get_global_id(0) * workGroupSize;"
                         "for(int j=0; j<workGroupSize; ++j)" \
                        " atomic_inc(&counts[offset + ((global uchar *)&input[offset2 + j])[pass]]);" \
                        "/*++_counts[((global uchar *)&input[get_global_id(0) * workGroupSize + j])[pass]];" \
                        "for(int j=0; j<256; ++j) counts[j] = _counts[j];*/" \
                        "}"}};

        auto sourcesById1 =
                std::map<std::string, std::string>{{"Count", "kernel void Count(global uint *const input, global uint * counts, int const pass) { "
                "/*local uint _counts[256];"
                "if(get_local_id(0) == 0) for(int j=0; j<256; ++j) _counts[j] = 0; barrier(CLK_LOCAL_MEM_FENCE);*/"
                "atomic_inc(&counts[get_group_id(0) * 256 + ((global uchar *)&input[get_global_id(0)])[pass]]);"
                "/*barrier(CLK_LOCAL_MEM_FENCE); if(get_local_id(0) == 0) for(int j=0; j<256; ++j) counts[get_group_id(0) * 256 + j] = _counts[j];*/"
                  "}"}};

        auto sourcesById =
                        std::map<std::string, std::string>{{"Count", "/*uint constant masks[4] = {0xFF, 0xFF00, 0xFF0000, 0xFF000000};*/"                        
                            "kernel void Count(global uint *const input, global uint *const input2, global uchar *const result, global uint *const counts, int const subGroupSize, int const pass) { "
                            "global uint *const _input = (*result & 2) ? input2 : input;"
                            "const size_t globalOffsetTgt = get_global_id(0) << 8;"
                            "const size_t globalOffsetSrc = get_global_id(0) << (8+4);"
                            "/*const size_t localOffset = get_local_id(0) << 8;*/"
                            "private uint _counts[256];"
                            "for(int j=0; j<256; ++j) _counts[j] = 0;/**/"
                            "/*const uint shift = pass << 3;"
                            "private uint const masks[4] = {0xFF, 0xFF00, 0xFF0000, 0xFF000000};*/"
                            "for(int j=0; j<256*16; ++j){"
                             "/*uint const value = _input[globalOffsetSrc + j];"
                            "uint const symbol = (value & masks[pass]) >> shift;"
                            "++_counts[symbol];*/"
                            " ++_counts[(((global uchar *)&_input[globalOffsetSrc + j])[pass])];"
                            "/* ++counts[globalOffsetTgt + (((global uchar *)&_input[globalOffsetSrc + j])[pass])];*/"
                            "}"
                            "for(int j=0; j<256; ++j){"
                            "   counts[globalOffsetTgt + j] = _counts[j];/**/"
                            "}"
                        "}"}};


        auto sourcesByIdSort = std::map<std::string, std::string>{
       {"Sort", "kernel void Sort(global uint *const keysOut, global uint *const counts, global uint *const offsets, global uint *const keysIn, global uchar *const result, uint const blockSize, uint const index){"
            "if ((*result & 1) == 0) return;"
            "global uint *const _output = (*result & 2) ? keysIn : keysOut;"
            "global uint *const _input = (*result & 2) ? keysOut : keysIn;"
            "const uint offset = get_global_id(0) * 256;"
            "local uint _offsets[256 * 2];"
            "{"
            "uint old = 0; uint current = 0;"
            "for(int symbol = 0; symbol < 256; ++symbol){"
                "old = current;"
                "current = old + counts[offset + symbol];"
                "_offsets[symbol + get_local_id(0) * 256] = old;"
            "}"
            "}"
            "const uint offset2 = get_global_id(0) * blockSize;"
            "local uint out[1<<12];"
            "for(int i=0; i<blockSize; ++i){"
                "uchar const symbol = ((global uchar*)&_input[offset2 + i])[index];"
                "out[_offsets[symbol + get_local_id(0) * 256]++ + (get_local_id(0) << 11)] = _input[offset2 + i];"
            "}"
            "uint old = 0;"
            "for(int symbol=0; symbol<256; ++symbol){"
                "const uint o = offsets[offset + symbol];"
                "for(int j=0; j<_offsets[symbol] - old; ++j){"
                    "_output[o + j] = out[old + j + (get_local_id(0) << 11)];"
                "}"
                "old = _offsets[symbol];"
            "}"
        "}"}};
        
        auto sourcesByIdExclusiveScan_CPU = std::map<std::string, std::string>{
       {"ExclusiveScan", "kernel void ExclusiveScan(global uint *const counts, global uint *const offsets, global uchar *const result, uint const blockCount){"
           "uint old = 0; uint current = 0; uint base = 0; int z = 0;"
            "for(int symbol = 0; symbol < 256; ++symbol){"
                "for(uint block=0; block<blockCount; ++block){"
                    "old = current;"
                    "current = old + counts[block * 256 + symbol];"
                    "offsets[block * 256 + symbol] = old;"
                "}"
                "/*if(base == current) ++z;"
                "else base = current;*/"
           "}"
           "/**result = (z != 255) ? (*result & 2 ? 1 : 3) : (*result & 2 ? 2 : 0);*/"
        "}"
       }};
         auto sourcesByIdExclusiveScan = std::map<std::string, std::string>{
       {"ExclusiveScan", "kernel void ExclusiveScan(global uint *const input, global uint *const input2, global uint *const counts, global uint *const offsets, global uchar *const result, uint const pass){"
           "private uint _offsets[256];"
           "const size_t globalId = get_global_id(0) << 8;"
           "_offsets[0] = 0;"
            "for(int symbol = 1; symbol < 256; ++symbol){"
                    "_offsets[symbol] = _offsets[symbol - 1] + counts[globalId + symbol - 1];"
            "}"
            "global uint *const _input = (*result & 2) ? input2 : input;"
            "private uint _output[1u<<12];"
            "const size_t globalOffsetSrc = get_global_id(0) << (8+4);"
            "for(int i=0; i<1u<<12; ++i){"
                "uint const value = _input[globalOffsetSrc + i];"
                "uint const symbol = (value & (0xFF << (pass * 8))) >> (pass * 8);"
                "_output[_offsets[symbol]++] = value;"
            "}"
            "/*if ((*result & 1) == 0) return;*/"
            "global uint *const output = (*result & 2) ? input : input2;"
            "uint offsetSymbol = 0;"
            "uint offsetSymbolSrc = 0;"
            "for(int symbol=0; symbol<256; ++symbol){"
                "for(uint j=0; j<get_global_id(0); ++j){"
                    "offsetSymbol += counts[j*256 + symbol];"
                "}"

                "for(uint j=0; j< counts[get_global_id(0)*256 + symbol]; ++j){"
                    "output[offsetSymbol++] = _output[0];"
                "}"
                "for(uint j=get_global_id(0)+1; j<get_global_size(0); ++j){"
                    "offsetSymbol += counts[j*256 + symbol];"
                "}"
            "}"
        "}"
       }};
         {
            auto sources = Ocl::Sources(sourcesById);
            auto sourcesSort = Ocl::Sources(sourcesByIdSort);
            auto sourcesExclusiveScan = Ocl::Sources(sourcesByIdExclusiveScan);
            auto context = Ocl::Context(devices);

            auto program = Ocl::Program(sources, context);
            auto programSort = Ocl::Program(sourcesSort, context);
            auto programExclusiveScan = Ocl::Program(sourcesExclusiveScan, context);
            for(auto &&device : devices){
                std::cout << "[Program]" << bool(program) << ": " << device << ": " << (Ocl::Handle<Ocl::Program<>>const&)program << ": " << Ocl::Info(program) << Ocl::Info(program, device, Query<CL_PROGRAM_BUILD_LOG>()) << "\n";
                std::cout << "[Program]" << bool(programSort) << ": " << device << ": " << (Ocl::Handle<Ocl::Program<>>const&)programSort << ": " << Ocl::Info(programSort) << Ocl::Info(programSort, device, Query<CL_PROGRAM_BUILD_LOG>()) << "\n";
                std::cout << "[Program]" << bool(programExclusiveScan) << ": " << device << ": " << (Ocl::Handle<Ocl::Program<>>const&)programExclusiveScan << ": " << Ocl::Info(programExclusiveScan) << Ocl::Info(programExclusiveScan, device, Query<CL_PROGRAM_BUILD_LOG>()) << "\n";
            }
            /*CPU*
            const size_t count = ((1 << 19) * 2);
            const uint32_t subGroupSize = 1 << 16;
            const auto workGroupCount = count / subGroupSize;
            const uint32_t workGroupSize = 1;
*/
            const uint32_t count = ((1 << 20));
            const uint32_t subGroupSize = 1 << 12;
            const uint32_t workGroupCount = count / subGroupSize;
            const uint32_t workGroupSize = 1;/**/

            auto deviceInput = Ocl::Memory(context, Ocl::SizeOf<cl_uint>(count), Query<CL_MEM_READ_WRITE>());
            auto deviceTarget = Ocl::Memory(context, Ocl::SizeOf<cl_uint>(count), Query<CL_MEM_READ_WRITE>());
            auto deviceCounts = Ocl::Memory(context, Ocl::SizeOf<cl_uint>((1u<< 8) * 256), Query<CL_MEM_READ_WRITE>());
            auto deviceOffsets = Ocl::Memory(context, Ocl::SizeOf<cl_uint>(subGroupSize*2 * 256), Query<CL_MEM_READ_WRITE>());
            auto deviceResult = Ocl::Memory(context, Ocl::SizeOf<cl_uchar>(1), Query<CL_MEM_READ_WRITE>());
            std::cout << "[Buffer]" << bool(deviceCounts) << ": " << (Ocl::Handle<Ocl::Memory<>>const&)deviceCounts << ": " << Ocl::Info(deviceCounts) << "\n";
            std::cout << "Program: CL_PROGRAM_NUM_KERNELS: " << Ocl::Info(program, Query<CL_PROGRAM_NUM_KERNELS>()) << "\n";
            std::cout << "CL_PROGRAM_SOURCE: " << Ocl::Info(program, Query<CL_PROGRAM_SOURCE>()) << "\n";
            std::cout << "CL_PROGRAM_KERNEL_NAMES: " << Ocl::Info(program, Query<CL_PROGRAM_KERNEL_NAMES>()) << "\n";
            std::cout << "Program: CL_PROGRAM_NUM_KERNELS: " << Ocl::Info(programSort, Query<CL_PROGRAM_NUM_KERNELS>()) << "\n";
            std::cout << "CL_PROGRAM_SOURCE: " << Ocl::Info(programSort, Query<CL_PROGRAM_SOURCE>()) << "\n";
            std::cout << "CL_PROGRAM_KERNEL_NAMES: " << Ocl::Info(programSort, Query<CL_PROGRAM_KERNEL_NAMES>()) << "\n";

            auto kernel = Ocl::Kernel(program, (std::string const&)Ocl::Info(program, Query<CL_PROGRAM_KERNEL_NAMES>()));
            std::cout << "[Kernel]" << bool(kernel) << ": " << (Ocl::Handle<Ocl::Kernel<>>const&)kernel << ": " << Ocl::Info(kernel) << "\n";
            auto kernelSort = Ocl::Kernel(programSort, (std::string const&)Ocl::Info(programSort, Query<CL_PROGRAM_KERNEL_NAMES>()));
            std::cout << "[Kernel]" << bool(kernelSort) << ": " << (Ocl::Handle<Ocl::Kernel<>>const&)kernelSort << ": " << Ocl::Info(kernelSort) << "\n";
            auto kernelExclusiveScan = Ocl::Kernel(programExclusiveScan, (std::string const&)Ocl::Info(programExclusiveScan, Query<CL_PROGRAM_KERNEL_NAMES>()));
            std::cout << "[Kernel]" << bool(kernelExclusiveScan) << ": " << (Ocl::Handle<Ocl::Kernel<>>const&)kernelExclusiveScan << ": " << Ocl::Info(kernelExclusiveScan) << "\n";
            
            auto j = 0;
            for(auto &&device : devices){
            //  if(++j==1)continue;
            /*
             for(auto &&_kernel : {kernel, kernelExclusiveScan, kernelSort}){
                std::cout << "CL_KERNEL_WORK_GROUP_SIZE: " << Ocl::Info(_kernel, device, Query<CL_KERNEL_WORK_GROUP_SIZE>()) << "\n";
                std::cout << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << Ocl::Info(_kernel, device, Query<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>()) << "\n";
                std::cout << "CL_KERNEL_LOCAL_MEM_SIZE: " << Ocl::Info(_kernel, device, Query<CL_KERNEL_LOCAL_MEM_SIZE>()) << "\n";
                std::cout << "CL_KERNEL_PRIVATE_MEM_SIZE: " << Ocl::Info(_kernel, device, Query<CL_KERNEL_PRIVATE_MEM_SIZE>()) << "\n";
             }*/
{
                auto commandQueue = context(device);
                std::cout << "[CommandQueue]" << bool(commandQueue) << device << ": " << (Ocl::Handle<Ocl::CommandQueue<>>const&)commandQueue << ": " << Ocl::Info(commandQueue) << "\n";      
                             
                uint32_t *result = nullptr;
                uint32_t *input = nullptr;
                cl_uint *counts = nullptr;
                cl_uchar *resultExcl = nullptr;
                commandQueue(deviceResult)(resultExcl);
                commandQueue(deviceInput)(input);

                clFinish(commandQueue);

                for(auto i=0; i<count; ++i)
                    input[i] = count - i;

                *resultExcl = 0;

                commandQueue(deviceInput)(std::move(input));
                commandQueue(deviceResult)(std::move(resultExcl));
                kernel(deviceInput, deviceTarget, deviceResult, deviceCounts, subGroupSize);
                kernelExclusiveScan(deviceInput, deviceTarget, deviceCounts, deviceOffsets, deviceResult, 256u);
                kernelSort(deviceInput, deviceCounts, deviceOffsets, deviceTarget, deviceResult, subGroupSize);
                clFinish(commandQueue);
                
      auto durationHist = Metrics::Duration<float, std::micro>();
      {
             auto integrator = Metrics::Integrator(durationHist);
                for(auto pass = 0; pass < 4; ++pass){
                
                   kernel(std::make_tuple(5, pass));
                   auto event1 = commandQueue(kernel)(Ocl::Size(1u << (8)), Ocl::Size(1u << 4));
                   kernelExclusiveScan(std::make_tuple(5, pass));
                //   auto
                 /*  event1 = commandQueue(kernelExclusiveScan)
                   (event1)
                   (Ocl::Size(1u<<8), Ocl::Size(1u << 0));*/
                   kernelSort(std::make_tuple(6, pass));
                 //  event1 = commandQueue(kernelSort)(event1)(Ocl::Size(workGroupCount), Ocl::Size(workGroupSize));
                   clWaitForEvents(1, (cl_event*)&event1);
              //     commandQueue(deviceCounts)(counts);
              //     clFinish(commandQueue);
              //     for(auto j=0;j<256;++j)
              //     std::cout << counts[j] << ",";
              //  std::cout << "\n";
              //     commandQueue(deviceCounts)(std::move(counts));
              //     clFinish(commandQueue);
               //    break;
               //    std::cout << "Pass: " << pass << "\n";
                }
                  auto event1 = commandQueue(deviceResult)(resultExcl);
                  clWaitForEvents(1, (cl_event*)&event1);
                  event1 = commandQueue(!(*resultExcl & 2) ? deviceTarget : deviceInput)(result);
                  clWaitForEvents(1, (cl_event*)&event1);
                }
                std::cout << "Duration: " << durationHist << "\n";
                for(auto i=0; i<10; ++i) std::cout << result[i] << ",";
                std::cout << "\n";
                commandQueue(!(*resultExcl & 2) ? deviceTarget : deviceInput)(std::move(result));

}           
             }
        }
     }

    std::cout << "Done.\n";

    return 0;
}


inline auto result = RadixSort(); 
}