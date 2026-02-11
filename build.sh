#!/bin/bash

export RAYNET_HOME=$HOME/raynet

# Usage info
show_help(){

echo {"Usage: ${0##*/} [-h] [-m BUILDMODE] [-f FEATURE]...
	Build (compile and link) Raynet with feature FEATURE in BUILDMODE mode. 

       -h            display this help and exit
	   -b			 make with 'bear -- make' to generate compile commands (requires bear)
	   -c			 clean simulation libarries before building
       -r            rebuild (deletes the current build directory before proceeding)
       -m BUILDMODE  chose between release and debug modes. Defaults to release.
       -f FEATURE    chose the feature to build. Currently available:
                          RLRDP (default) Builds Raynet to support RDP Agents
                          CARTPOLE        Builds raynet for cartpole experimentation
                          ORCA            Builds raynet for ORCA experimentation
                          JAMESCC         Build raynet for JamesCC basic congestion control examples"
}
   
   # Initialize our own variables:
   mode="release"
   feature="CARTPOLE"
   make_prefix=""
   clean="false"
   rebuild="false"

   OPTIND=1
   # Resetting OPTIND is necessary if getopts was used previously in the script.
   # It is a good idea to make OPTIND local if you process options in a function.
   
   while getopts hbcrm:f: opt; do
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
if [ "$mode" != "release" ] && [ "$mode" != "debug" ] 
then
	echo "-m option value not recognised. Select between release and debug"
	echo "Build failed."	
	exit 1 
	fi

# Check for invalid feature #todo: remove this? New users should not need to modify the build script to get their feature built
if [ "$feature" != "CARTPOLE" ] && [ "$feature" != "ORCA" ] && [ "$feature" != "JAMESCC" ]
then
	echo "-f option value not recognised. Select among CARTPOLE or ORCA"
	echo "Build failed."	
	exit 1  
	fi
export RAYNET_FEATURE=$feature

# --------------------------------------------------------------------------------------------

simlibs=("$RAYNET_HOME/simlibs/RLComponents"
         "$RAYNET_HOME/simlibs/TcpPaced")

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
    if [ ${exists} = 0 ] 
    then
        echo "${dir##*/}"    # print everything after the final "/"
        simlibs=("${simlibs[@]}" ${dir})
    fi
done

echo "simlibs"
echo ${simlibs[@]}
# TODO: Loop through all simlibs and append their paths to this list automatically
# projects=(
#         "$RAYNET_HOME/simlibs/RLComponents"
#         "$RAYNET_HOME/simlibs/TcpPaced"
#         "$RAYNET_HOME/simlibs/cartpole"
#         "$RAYNET_HOME/simlibs/JamesLib"
#     )
# echo "projects"
# echo ${projects[@]}

for simlib in "${simlibs[@]}"; do
    echo ${simlib}
done


# (Optional) Clean simulation libraries
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

# Build simulation libraries
if [ "$mode" = "release" ]; then
    
    echo "Building INET release..."
    echo "---------------------------------"
    cd "$HOME/inet4.5" && make -j32 MODE=release
    echo "---------------------------------"
    echo ""

    for proj in "${simlibs[@]}"; do
        echo "Building release libraries in $proj..."
        echo "---------------------------------"
        cd "$proj" || exit 1
        ${make_prefix}make makefilesrelease     # Generate makefile in the simlib src directory using opp_makemake
        ${make_prefix}make -j32 MODE=release    # Bulid simlib with the generated makefile
        echo "---------------------------------"
        echo ""
    done
fi

if [ "$mode" = "debug" ]; then
    echo "Building INET debug..."
    echo "---------------------------------"
    cd "$HOME/inet4.5" && make -j32 MODE=debug
    echo "---------------------------------"
    echo ""

    for proj in "${simlibs[@]}"; do
        echo ""
        echo "Building debug libraries in $proj..."
        cd "$proj" || exit 1
        ${make_prefix}make makefilesdebug
        ${make_prefix}make -j32 MODE=debug
    done
fi

# Build raynet
cd $RAYNET_HOME
if [ "$rebuild" = "true" ]; then
    rm -r build
fi
mkdir build 
cd build

echo "Building RayNet..."
echo "---------------------------------"
cmake -DCMAKE_BUILD_TYPE=$mode -DCLEAN_ALL=$clean ../ && \
make -j32
echo "---------------------------------"