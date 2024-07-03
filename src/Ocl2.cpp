//
// File:       hello.c
//
// Abstract:   A simple "Hello World" compute example showing basic usage of OpenCL which
//             calculates the mathematical square (X[i] = pow(X[i],2)) for a buffer of
//             floating point values.
//             
//
// Version:    <1.0>
//
// Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple Inc. ("Apple")
//             in consideration of your agreement to the following terms, and your use,
//             installation, modification or redistribution of this Apple software
//             constitutes acceptance of these terms.  If you do not agree with these
//             terms, please do not use, install, modify or redistribute this Apple
//             software.
//
//             In consideration of your agreement to abide by the following terms, and
//             subject to these terms, Apple grants you a personal, non - exclusive
//             license, under Apple's copyrights in this original Apple software ( the
//             "Apple Software" ), to use, reproduce, modify and redistribute the Apple
//             Software, with or without modifications, in source and / or binary forms;
//             provided that if you redistribute the Apple Software in its entirety and
//             without modifications, you must retain this notice and the following text
//             and disclaimers in all such redistributions of the Apple Software. Neither
//             the name, trademarks, service marks or logos of Apple Inc. may be used to
//             endorse or promote products derived from the Apple Software without specific
//             prior written permission from Apple.  Except as expressly stated in this
//             notice, no other rights or licenses, express or implied, are granted by
//             Apple herein, including but not limited to any patent rights that may be
//             infringed by your derivative works or by other works in which the Apple
//             Software may be incorporated.
//
//             The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
//             WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
//             WARRANTIES OF NON - INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
//             PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION
//             ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
//
//             IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
//             CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//             SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//             INTERRUPTION ) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
//             AND / OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER
//             UNDER THEORY OF CONTRACT, TORT ( INCLUDING NEGLIGENCE ), STRICT LIABILITY OR
//             OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright ( C ) 2008 Apple Inc. All Rights Reserved.
//

////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include <cstdint>
#include<iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include "Ocl.hpp"

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (1024)

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array 
//
const char *KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i]*input[i];                                \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////
template<auto>
class Query{};
ocl_test()
{
    int err;                            // error code returned from api calls
      
    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    // Fill our data set with random float values
    //

    for(auto&&platform : Ocl::Enum(Ocl::Platform())){
        auto devices =  Ocl::Enum(Ocl::Device(), platform);
      //
    auto context = Ocl::Context(devices);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }


  auto input = Ocl::Memory(context, Ocl::SizeOf<float>(DATA_SIZE), Query<CL_MEM_READ_ONLY>());
  auto output =  Ocl::Memory(context, Ocl::SizeOf<float>(DATA_SIZE), Query<CL_MEM_WRITE_ONLY>());
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    auto sourcesById = std::map<std::string, std::string>{{"kernel", KernelSource}};
    auto sources = Ocl::Sources(sourcesById);
    auto program = Ocl::Program(sources, context);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    auto kernel = Ocl::Kernel(program, "square");
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    unsigned int const count = DATA_SIZE;

     kernel(input, output, count);
    
    for(auto&&device : devices){
        std::cout << Ocl::Info(program, device, Query<CL_PROGRAM_BUILD_LOG>()) << "\n";
        auto commands = context(device);
        if (!commands)
        {
            printf("Error: Failed to create a command commands!\n");
            return EXIT_FAILURE;
        }
        float *input_host = nullptr;
        auto event1 = commands(input)(input_host); //map to host
        float *result_host = nullptr;
        auto event2 = commands(output)(result_host);
        clWaitForEvents(1, (cl_event *)&event1);
        for(auto i = 0; i < count; i++){
            input_host[i] = rand() / (float)RAND_MAX;
        }
       //  std::cout << std::uintptr_t(input_host) << ":" <<  std::uintptr_t(result_host) << "\n";
        event1 = commands(input)();
        event1 = commands(kernel)(event1, event2)(Ocl::Size(count));
        event1 = commands(output)(event1)(CL_MIGRATE_MEM_OBJECT_HOST);
        clFinish(commands);
        correct = 0;
        for(auto i = 0; i < count; i++)
        {
            if(result_host[i] == input_host[i] * input_host[i])
                correct++;
        }
    commands(input)(std::move(input_host));
    commands(output)(std::move(result_host));

    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values!\n", correct, count);
       }
    }
    return 0;
}

inline auto res = ocl_test();