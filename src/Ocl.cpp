#include <iostream>
#include <cstdint>

#include "Ocl.hpp"

#define MT(x) std::make_tuple(#x, Query<x>())

template<auto type>
void Test(cl_event event, cl_int event_command_status, void *user_data){
    std::cout << "Event: " << type << ":" << event << " " << event_command_status << "\n";
}


template<auto type>
void Test2(cl_event event, cl_int event_command_status, void *user_data){
    std::cout << "Result2: " << *(int32_t*)user_data << "\n";
}

template<auto type>
void Test3(cl_event event, cl_int event_command_status, void *user_data){
    std::cout << "Result3: " << std::uintptr_t(*(int32_t**)user_data) << "\n";
}

#include <chrono>
#include <thread>
template<auto>class Query{};
int32_t EnumeratePlatforms(){
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
                                                     MT(CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), MT(CL_DEVICE_MAX_MEM_ALLOC_SIZE), MT(CL_DEVICE_GLOBAL_MEM_SIZE), MT(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE), MT(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE), MT(CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE), MT(CL_DEVICE_LOCAL_MEM_SIZE),
                                                     std::make_tuple("CL_DEVICE_SVM_CAPABILITIES", Query<CL_DEVICE_SVM_CAPABILITIES>()), MT(CL_DEVICE_OPENCL_C_VERSION));
            std::apply([&device](auto&&... query) {((std::cout << "\t\t" << std::get<0>(query)<< ": " << Ocl::Info(device, std::get<1>(query)) << '\n'), ...);}, queries);
        }
        auto sourcesById =
                        //std::map<std::string, std::string>{{"nop", "kernel void nop() {}"}};
                        std::map<std::string, std::string>{{"buffer", "kernel void buffer(global int * const result) {*result = 9999;}"}};
        
        {
            auto sources = Ocl::Sources(sourcesById);
            auto context = Ocl::Context(devices);

            auto program = Ocl::Program(sources, context);
            for(auto &&device : devices)
                std::cout << "[Program]" << bool(program) << ": " << device << ": " << (Ocl::Handle<Ocl::Program<>>const&)program << ": " << Ocl::Info(program) << Ocl::Info(program, device, Query<CL_PROGRAM_BUILD_LOG>()) << "\n";
            
            auto deviceBuffer = Ocl::Memory(context, Ocl::SizeOf<int32_t>(), Query<CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY>());
            
            std::cout << "[Buffer]" << bool(deviceBuffer) << ": " << (Ocl::Handle<Ocl::Memory<>>const&)deviceBuffer << ": " << Ocl::Info(deviceBuffer) << "\n";
            std::cout << "Program: CL_PROGRAM_NUM_KERNELS: " << Ocl::Info(program, Query<CL_PROGRAM_NUM_KERNELS>()) << "\n";
            std::cout << "CL_PROGRAM_SOURCE: " << Ocl::Info(program, Query<CL_PROGRAM_SOURCE>()) << "\n";
            std::cout << "CL_PROGRAM_KERNEL_NAMES: " << Ocl::Info(program, Query<CL_PROGRAM_KERNEL_NAMES>()) << "\n";

            auto kernel = Ocl::Kernel(program, (std::string const&)Ocl::Info(program, Query<CL_PROGRAM_KERNEL_NAMES>()));
            std::cout << "[Kernel]" << bool(kernel) << ": " << (Ocl::Handle<Ocl::Kernel<>>const&)kernel << ": " << Ocl::Info(kernel) << "\n";
            //std::cout << "Arg: " << clSetKernelArg(kernel, 0, sizeof(cl_mem), &handle) << "\n";
            kernel(deviceBuffer);
            for(auto &&device : devices){
                std::cout << "CL_KERNEL_WORK_GROUP_SIZE: " << Ocl::Info(kernel, device, Query<CL_KERNEL_WORK_GROUP_SIZE>()) << "\n";
                std::cout << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << Ocl::Info(kernel, device, Query<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>()) << "\n";
                std::cout << "CL_KERNEL_LOCAL_MEM_SIZE: " << Ocl::Info(kernel, device, Query<CL_KERNEL_LOCAL_MEM_SIZE>()) << "\n";
                std::cout << "CL_KERNEL_PRIVATE_MEM_SIZE: " << Ocl::Info(kernel, device, Query<CL_KERNEL_PRIVATE_MEM_SIZE>()) << "\n";
{
                auto commandQueue = context(device);
                std::cout << "[CommandQueue]" << bool(commandQueue) << device << ": " << (Ocl::Handle<Ocl::CommandQueue<>>const&)commandQueue << ": " << Ocl::Info(commandQueue) << "\n";      
             
                auto event0 = commandQueue(deviceBuffer)(CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED)(CL_COMPLETE, Test<CL_COMPLETE>);
             
                //auto &&[hostMem, event] = Ocl::Memory(commandQueue, deviceBuffer, Map<int>());
                //auto &&[memoryMap, event] = commandQueue(deviceBuffer)(event0)(Ocl::Host<int>());
                
                int32_t *result = nullptr;
                auto event1 = commandQueue(deviceBuffer)(event0)(result)(CL_COMPLETE, Test3<CL_COMPLETE>, &result); //map to host
                {
                    auto event = commandQueue(kernel)(event0)(Ocl::Size(1u))(CL_COMPLETE, Test<CL_COMPLETE>);
                    auto event2 = commandQueue(kernel)(event)(Ocl::Size(1u), Ocl::Size(1u))(CL_COMPLETE, Test<CL_COMPLETE>);
                    auto event3 = commandQueue(kernel)(event2)(std::array{Ocl::Size(0u), Ocl::Size(1u)}, Ocl::Size(1u))(CL_COMPLETE, Test<CL_COMPLETE>);
                   
                   event3 = commandQueue(deviceBuffer)(event1, event2)(CL_MIGRATE_MEM_OBJECT_HOST)(CL_COMPLETE, Test2<CL_COMPLETE>, result);
                
                commandQueue(deviceBuffer)(event3)(std::move(result));
                }
}           
             }
        }
     }

    std::cout << "Done.\n";

    return 0;
}


inline auto result = EnumeratePlatforms();