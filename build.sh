#!/bin/bash

export RAYNET_HOME=$HOME/raynet

# Usage info
show_help(){

echo {"Usage: ${0##*/} [-h] [-m BUILDMODE]...
	Build (compile and link) Raynet BUILDMODE mode. 

       -h            display this help and exit
	   -b			 make with 'bear -- make' to generate compile commands (requires bear)
	   -c			 clean simulation libarries before building
       -r            rebuild (deletes the current build directory before proceeding)
       -m BUILDMODE  chose between release and debug modes. Defaults to release."
}
   
   # Initialize our own variables:
   mode="release"
   make_prefix=""
   clean="false"
   rebuild="false"

   OPTIND=1
   # Resetting OPTIND is necessary if getopts was used previously in the script.
   # It is a good idea to make OPTIND local if you process options in a function.
   
   while getopts hbcrm: opt; do
       case $opt in
           h)
               show_help
               exit 0
               ;;
           b)  
		   	   make_prefix="bear -- "
			   ;;
           c)
		       clean='true'
			   ;;
           r)
               rebuild='true'
               ;;
           m)  mode=$OPTARG
               ;;
			   
           *)
               show_help >&2
               exit 1
               ;;
       esac
   done
   shift "$((OPTIND-1))"   # Discard the options and sentinel --

# Check for invalid build mode
if [ "$mode" != "release" ] && [ "$mode" != "debug" ] 
then
	echo "-m option value not recognised. Select between release and debug"
	echo "Build failed."	
	exit 1 
	fi

# Command/flags have been validated. Begin building here.
# ----------------------------------------------------------------------------------------------------------------------------------

source $HOME/omnetpp/setenv

# List of simlibs. Any simlib that needs to be compiled FIRST (eg. is a dependency) should be added here, rather than the loop.
simlibs=("$RAYNET_HOME/simlibs/RLComponents"
"$RAYNET_HOME/simlibs/tcpPacedNoCC"
)

# list directories in the form "/tmp/dirname/"
for dir in $RAYNET_HOME/simlibs/*/     
do
    dir=${dir%*/}      # remove the trailing "/"
    exists=0
    for simlib in "${simlibs[@]}" 
    do
        if [ "${simlib}" = "${dir}" ] 
        then
            exists=1
        fi
    done
    if [ ${exists} = 0 ] # Only append the simlib to the list if it was not previously included
    then
        simlibs=("${simlibs[@]}" ${dir})    # Append the simulation library to the simlibs list
    fi
done

# Debug print:
echo "Simulation libraries detected:"
echo ${simlibs[@]}
for simlib in "${simlibs[@]}"; do
    echo ${simlib}
done

# -c flag: Clean simulation libraries
if [ "$clean" = "true" ]; then
    echo "Cleaning simulation libararies..."
    echo ""

    for proj in "${simlibs[@]}"; do
        echo "Cleaning $proj..."
        echo "---------------------------------"
        cd "$proj" || exit 1
        ${make_prefix}make cleanall
        echo "---------------------------------"
        echo ""
    done
    echo "Cleaning complete!"
    echo ""
fi

# Build INET (release or debug)
echo "Building INET $mode..."
echo "---------------------------------"
cd "$HOME/inet4.5" && make -j32 MODE=$mode
echo "---------------------------------"
echo ""

# Build simulation libraries (release or debug)
for proj in "${simlibs[@]}"; do
    echo "Building $mode libraries in $proj..."
    echo "---------------------------------"
    cd "$proj" || exit 1
    # ${make_prefix}make makefiles$mode       # Generate makefile in the simlib src directory using opp_makemake
    ${make_prefix}make makefiles MODE=$mode   # Just in case the simlib only has a make makefiles option, and no make makefilesrelease or make makefilesdebug (cubic does this)
    ${make_prefix}make -j32 MODE=$mode      # Build simlib with the generated makefile
    echo "---------------------------------"
    echo ""
done

# -r flag: Remove any existing build directory before building
if [ "$rebuild" = "true" ]; then  
    cd $RAYNET_HOME
    rm -r build 
fi

# Build raynet (release or debug)
cd $RAYNET_HOME
mkdir build 
cd build
echo "Building RayNet..."
echo "---------------------------------"
cmake -DCMAKE_BUILD_TYPE=$mode ../ && \
make -j32
echo "---------------------------------"