#!/bin/bash

export RAYNET_HOME=$HOME/raynet

# Usage info
show_help(){

echo {"Usage: ${0##*/} [-h] [-m BUILDMODE] [-f FEATURE]...
	Build (compile and link) Raynet with feature FEATURE in BUILDMODE mode. 

       -h            display this help and exit
	   -b			 make with 'bear -- make' to generate compile commands (requires bear)
	   -c			 clean simulation libarries before building
       -m BUILDMODE  chose between release and debug modes. Defaults to release.
       -f FEATURE    chose the feature to build. Currently available:
                          RLRDP (default) Builds Raynet to support RDP Agents
                          CARTPOLE        Builds raynet for cartpole experimentation
                          ORCA            Builds raynet for ORCA experimentation"
}
   
   # Initialize our own variables:
   mode="release"
   feature="CARTPOLE"
   make_prefix=""
   clean=""

   OPTIND=1
   # Resetting OPTIND is necessary if getopts was used previously in the script.
   # It is a good idea to make OPTIND local if you process options in a function.
   
   while getopts hbcm:f: opt; do
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
           m)  mode=$OPTARG
               ;;
           f)  feature=$OPTARG
               ;;
			   
           *)
               show_help >&2
               exit 1
               ;;
       esac
   done
   shift "$((OPTIND-1))"   # Discard the options and sentinel --

# Check for invalid build mode
if [ "$mode" != "debug" ] && [ "$mode" != "release" ]
then
	echo "-m option value not recognised. Select between release and debug"
	echo "Build failed."	
	exit 1 
	fi

# Check for invalid feature
if [ "$feature" != "CARTPOLE" ] && [ "$feature" != "ORCA" ]
then
	echo "-f option value not recognised. Select among CARTPOLE or ORCA"
	echo "Build failed."	
	exit 1  
	fi
export RAYNET_FEATURE=$feature

# --------------------------------------------------------------------------------------------

projects=(
        "$RAYNET_HOME/simlibs/RLComponents"
        "$RAYNET_HOME/simlibs/TcpPaced"
        "$RAYNET_HOME/simlibs/cartpole"
        "$RAYNET_HOME/simlibs/JamesLib"
    )

# (Optional) Clean simulation libraries
if [ "$clean" = "true" ]; then
    echo "Cleaning simulation libararies..."
    echo ""

    for proj in "${projects[@]}"; do
        echo "Cleaning $proj..."
        echo "---------------------------------"
        cd "$proj" || exit 1
        ${make_prefix}make cleanall
        echo "---------------------------------"
        echo ""
    done
fi

# Build simulation libraries
if [ "$mode" = "release" ]; then
    
    echo "Building INET release..."
    echo "---------------------------------"
    cd "$HOME/inet4.4" && make -j32 MODE=release
    echo "---------------------------------"
    echo ""

    for proj in "${projects[@]}"; do
        echo "Building release libraries in $proj..."
        echo "---------------------------------"
        cd "$proj" || exit 1
        ${make_prefix}make makefilesrelease
        ${make_prefix}make -j32 MODE=release
        echo "---------------------------------"
        echo ""
    done
fi

if [ "$mode" = "debug" ]; then
    echo "Building INET debug..."
    echo "---------------------------------"
    cd "$HOME/inet4.4" && make -j32 MODE=debug
    echo "---------------------------------"
    echo ""

    for proj in "${projects[@]}"; do
        echo ""
        echo "Building debug libraries in $proj..."
        cd "$proj" || exit 1
        ${make_prefix}make makefilesdebug
        ${make_prefix}make -j32 MODE=debug
    done
fi

# Build raynet
cd $RAYNET_HOME
mkdir build 
cd build

echo "Building RayNet..."
echo "---------------------------------"
if [ "$mode" = "release" ]
then
	cmake -DCMAKE_BUILD_TYPE=Release ../ && \
	make -j32
fi


if [ "$mode" = "debug" ]
then
	cmake -DCMAKE_BUILD_TYPE=Debug ../ && \
	make -j32
fi
echo "---------------------------------"


