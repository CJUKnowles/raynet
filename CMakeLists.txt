cmake_policy(SET CMP0048 NEW)
cmake_minimum_required(VERSION 3.5...3.18)
project(rlomnetbind VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_DEBUG_POSTFIX _dbg)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# For each simlib in the simlib directory, add it to link libraries
file(GLOB SIMLIB_DIRS RELATIVE ${PROJECT_SOURCE_DIR}/simlibs ${PROJECT_SOURCE_DIR}/simlibs/*)
foreach(lib ${SIMLIB_DIRS})
    if (IS_DIRECTORY ${PROJECT_SOURCE_DIR}/simlibs/${lib}/src)
        # Append library name to LIB_NAMES (used for target_link_libraries)
        list(APPEND LIB_NAMES ${lib})

        # Append library path to LIB_PATHS (used for link_directories)
        set(LIB_PATH ${PROJECT_SOURCE_DIR}/simlibs/${lib}/src)
        list(APPEND LIB_PATHS ${LIB_PATH})
        list(APPEND LIB_PATHS " ")

        # Set the path for this specific library (may not be needed)
        string(TOUPPER ${lib} LIB_NAME)
        set(${LIB_NAME}_PATH ${LIB_PATH})
        message(STATUS "${LIB_NAME}_PATH = ${${LIB_NAME}_PATH}")
    endif()
endforeach()

# External libraries (no looping here, these are required to make the project work)
set(INET_MODEL_PATH $ENV{HOME}/inet4.5/src)

set(TCPGOODPUT_PATH $ENV{HOME}/omnetpp/samples/tcpGoodputApplications/src) # tcpGoodputApplications, creates goodput vectors for plotting via custom TcpApp
list(APPEND LIB_PATHS ${TCPGOODPUT_PATH})
list(APPEND LIB_NAMES tcpGoodputApplications)

set(LEOSATELLITES_PATH $ENV{HOME}/omnetpp/samples/leosatellites/src) # leosatellites, not strictly necessary but contains features that ORBTCPEXPERIMENTS needs
list(APPEND LIB_PATHS ${LEOSATELLITES_PATH})
list(APPEND LIB_NAMES leosatellites)

set(ORBTCPEXPERIMENTS_PATH $ENV{HOME}/omnetpp/samples/orbtcpExperiments/src) # orbtcpExperiments, Aiden's experiment suite with some custom components like BandwidthRecorderDropTailQueue
list(APPEND LIB_PATHS ${ORBTCPEXPERIMENTS_PATH})
list(APPEND LIB_NAMES orbtcpExperiments)


# Declare Release libraries
set(MY_LIB_LINK_LIBRARIES_RELEASE optimized oppsim optimized oppcommon optimized oppcmdenv optimized Cmdrlenv optimized oppnedxml optimized oppenvir optimized -Wl,--no-as-needed INET) # Simlibs are automated now! But for reference: optimized -Wl,--no-as-needed TcpPaced optimized -Wl,--no-as-needed RLComponents optimized -Wl,--no-as-needed cartpole optimized -Wl,--no-as-needed JamesLib
message(MY_LIB_LINK_LIBRARIES_RELEASE IS ${MY_LIB_LINK_LIBRARIES_RELEASE})
foreach(lib_name ${LIB_NAMES})
    list(APPEND MY_LIB_LINK_LIBRARIES_RELEASE "optimized")
    list(APPEND MY_LIB_LINK_LIBRARIES_RELEASE "-Wl,--no-as-needed")
    list(APPEND MY_LIB_LINK_LIBRARIES_RELEASE "${lib_name}")
endforeach()
message(MY_LIB_LINK_LIBRARIES_RELEASE IS ${MY_LIB_LINK_LIBRARIES_RELEASE})

# Declare debug libraries
set(MY_LIB_LINK_LIBRARIES_DEBUG debug oppsim_dbg debug oppcommon_dbg debug oppcmdenv_dbg debug Cmdrlenv debug oppnedxml_dbg debug oppenvir_dbg debug -Wl,--no-as-needed INET_dbg) # Simlibs are automated now! But for reference: optimized -Wl,--no-as-needed TcpPaced optimized -Wl,--no-as-needed RLComponents optimized -Wl,--no-as-needed cartpole optimized -Wl,--no-as-needed JamesLib
message(MY_LIB_LINK_LIBRARIES_DEBUG IS ${MY_LIB_LINK_LIBRARIES_DEBUG})
foreach(lib_name ${LIB_NAMES})
    list(APPEND MY_LIB_LINK_LIBRARIES_DEBUG "debug")
    list(APPEND MY_LIB_LINK_LIBRARIES_DEBUG "-Wl,--no-as-needed")
    list(APPEND MY_LIB_LINK_LIBRARIES_DEBUG "${lib_name}_dbg")
endforeach()
message(MY_LIB_LINK_LIBRARIES_DEBUG IS ${MY_LIB_LINK_LIBRARIES_DEBUG})

# Link omnet
find_package(OmnetPP 6.0.1 REQUIRED)
include(OmnetppHelpers)
add_compile_options(-fpermissive -fno-omit-frame-pointer -fPIC -MMD -MP -MF)
add_compile_definitions(HAVE_SWAPCONTEXT)
find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/extern/pybind11/include ${OMNETPP_INCLUDE_DIR} ${OMNETPP_ROOT}/include ${OMNETPP_ROOT}/src ${LIB_PATHS})
link_directories(${OMNETPP_ROOT}/lib ${INET_MODEL_PATH} ${LIB_PATHS})

# Build Raynet
add_library(rlomnet SHARED ${PROJECT_SOURCE_DIR}/src/GymApi.cc)
add_library(Cmdrlenv SHARED ${PROJECT_SOURCE_DIR}/src/Cmdrlenv.cc)
add_executable(testcpp ${PROJECT_SOURCE_DIR}/src/gymApiTest.cc)

add_subdirectory(extern/pybind11)
pybind11_add_module(omnetbind MODULE ${PROJECT_SOURCE_DIR}/src/GymApiBind.cc)
set_target_properties(omnetbind PROPERTIES DEBUG_POSTFIX "") # Force omnetbind.so to not have the _dbg postfix. This allows the python import to continue working without changes.
message(PROJECT_SOURCE_DIR="${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL release)
    target_link_libraries(Cmdrlenv oppcmdenv)
    target_link_libraries(omnetbind PRIVATE ${MY_LIB_LINK_LIBRARIES_RELEASE})
    target_link_libraries(rlomnet PRIVATE ${MY_LIB_LINK_LIBRARIES_RELEASE})
    target_link_libraries(omnetbind PRIVATE rlomnet)
    target_link_libraries(testcpp rlomnet oppsim)
elseif(CMAKE_BUILD_TYPE STREQUAL debug)
    target_link_libraries(Cmdrlenv oppcmdenv_dbg)
    set_target_properties(Cmdrlenv PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    target_link_libraries(omnetbind PRIVATE rlomnet)
    target_link_libraries(rlomnet PRIVATE ${MY_LIB_LINK_LIBRARIES_DEBUG})
    set_target_properties(testcpp PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    target_link_libraries(testcpp rlomnet oppsim_dbg)
endif()