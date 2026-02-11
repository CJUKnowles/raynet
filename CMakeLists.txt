cmake_policy(SET CMP0048 NEW)
cmake_minimum_required(VERSION 3.4...3.18)
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

# Non-simlib libraries (no looping here, these are required to make the project work)
set(INET_MODEL_PATH $ENV{HOME}/inet4.5/src)

# Declare Release and Debug libraries
set(MY_LIB_LINK_LIBRARIES_RELEASE optimized oppsim optimized oppcommon optimized oppcmdenv optimized Cmdrlenv optimized oppnedxml optimized oppenvir optimized -Wl,--no-as-needed INET) # Simlibs are automated now! But for reference: optimized -Wl,--no-as-needed TcpPaced optimized -Wl,--no-as-needed RLComponents optimized -Wl,--no-as-needed cartpole optimized -Wl,--no-as-needed JamesLib
message(MY_LIB_LINK_LIBRARIES_RELEASE IS ${MY_LIB_LINK_LIBRARIES_RELEASE})
foreach(lib_name ${LIB_NAMES})
    list(APPEND MY_LIB_LINK_LIBRARIES_RELEASE "optimized")
    list(APPEND MY_LIB_LINK_LIBRARIES_RELEASE "-Wl,--no-as-needed")
    list(APPEND MY_LIB_LINK_LIBRARIES_RELEASE "${lib_name}")
endforeach()
message(MY_LIB_LINK_LIBRARIES_RELEASE IS ${MY_LIB_LINK_LIBRARIES_RELEASE})

# TODO: Repeat release link libarary looping logic for debug libaries
set(MY_LIB_LINK_LIBRARIES_DEBUG debug oppsim_dbg debug oppcommon_dbg debug oppcmdenv_dbg debug Cmdrlenv debug oppnedxml_dbg debug oppenvir_dbg debug -Wl,--no-as-needed INET_dbg debug -Wl,--no-as-needed TcpPaced_dbg debug -Wl,--no-as-needed TcpPaced_dbg debug -Wl,--no-as-needed RLComponents_dbg debug -Wl,--no-as-needed cartpole_dbg debug -Wl,--no-as-needed JamesLib_dbg)

# Link omnet
find_package(OmnetPP 6.0.1 REQUIRED)
include(OmnetppHelpers)
add_compile_options(-fpermissive -fno-omit-frame-pointer -fPIC -MMD -MP -MF )
add_compile_definitions($ENV{RAYNET_FEATURE} HAVE_SWAPCONTEXT  WITH_NETBUILDER)
find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/extern/pybind11/include ${OMNETPP_INCLUDE_DIR} ${OMNETPP_ROOT}/include ${OMNETPP_ROOT}/src ${LIB_PATHS})
link_directories(${OMNETPP_ROOT}/lib ${INET_MODEL_PATH} ${LIB_PATHS})

# Build Raynet
add_library(rlomnet SHARED ${PROJECT_SOURCE_DIR}/src/GymApi.cc)
add_library(Cmdrlenv SHARED ${PROJECT_SOURCE_DIR}/src/Cmdrlenv.cc)
add_executable(testcpp ${PROJECT_SOURCE_DIR}/src/gymApiTest.cc)

add_subdirectory(extern/pybind11)
pybind11_add_module(omnetbind MODULE ${PROJECT_SOURCE_DIR}/src/GymApiBind.cc)
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

    target_link_libraries(rlomnet PRIVATE ${MY_LIB_LINK_LIBRARIES_DEBUG})
    set_target_properties(testcpp PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    target_link_libraries(testcpp rlomnet oppsim_dbg)
endif()


