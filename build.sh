#!/bin/bash


check_dir() {
    local name="$1"
    local path="$2"

    if [[ -d "$path" ]]; then
        echo -e "$name:\t $path"
    else
        echo -e "$name:\t !!! path does not exist: $path !!!"
        echo "----------------------------------------------------------------------------------------------------------------"
        echo -e "ERROR: $path does not exist. Please check your $name environment variable in raynet/build_paths.sh"
        echo "Build failed."
        exit 1
    fi
}
echo ""
echo ""
echo "RayNet: Verifying critical paths..."
echo "----------------------------------------------------------------------------------------------------------------"
source "./build_paths.sh" # Contains user-defined build paths.
check_dir "RAYNET_PATH" "$RAYNET_PATH"
check_dir "OMNET_PATH" "$OMNET_PATH"
check_dir "INET_PATH" "$INET_PATH"
echo "----------------------------------------------------------------------------------------------------------------"
echo "Critical paths are valid. Proceeding..."
echo ""

# Usage info
show_help(){

echo {"Usage: ${0##*/} [-h] [-m BUILDMODE]...
	Build (compile and link) Raynet BUILDMODE mode. 

       -h            display this help and exit
	   -b			 make with 'bear -- make' to generate compile commands (requires bear)
	   -c			 clean simulation libarries before building
       -r            rebuild (deletes the current build directory before proceeding)
       -i            initialize RayNet Python virtual environment before building (calls create-venv.sh)
       -m BUILDMODE  chose between release and debug modes. Defaults to release."
}
   
   # Initialize our own variables:
   mode="release"
   make_prefix=""
   clean="false"
   rebuild="false"
   initialize="false"

   OPTIND=1
   # Resetting OPTIND is necessary if getopts was used previously in the script.
   # It is a good idea to make OPTIND local if you process options in a function.
   
   while getopts hbcrim: opt; do
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
           i)
               initialize='true'
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

# -i flag: Initialize python virtual environment
if [ "$initialize" = "true" ]; then
    echo "Initializing Python virtual environment..."
    echo "----------------------------------------------------------------------------------------------------------------"

    # Verify python3.12 exists
    if ! command -v python3.12 >/dev/null 2>&1; then
        echo "ERROR: python3.12 not found."
        echo "Please install Python 3.12 and try again."
        exit 1
    fi

    # Create virtual environment (raynet/.venv) and install dependencies
    cd "$RAYNET_PATH" || exit 1
    python3.12 -m venv .venv || exit 1
    source .venv/bin/activate || exit 1
    pip install --upgrade pip || exit 1
    pip install -r requirements-extra.txt || exit 1

    echo ""
    echo "----------------------------------------------------------------------------------------------------------------"
    echo "Virtual environment successfully initialized. Activate it before running RayNet:"
    echo "    source .venv/bin/activate"
    echo ""
    echo ""
fi

# ----------------------------------------------------------------------------------------------------------------
# Verify virtual environment exists
# ----------------------------------------------------------------------------------------------------------------
echo "RayNet: Verifying Python virtual environment..."
echo "----------------------------------------------------------------------------------------------------------------"
if [ ! -d "$RAYNET_VENV_PATH" ]; then
    echo -e "RAYNET_VENV_PATH:\t !!! path does not exist: $RAYNET_VENV_PATH !!!"
    echo "----------------------------------------------------------------------------------------------------------------"
    echo -e "ERROR: $RAYNET_VENV_PATH does not exist. Please check your RAYNET_VENV_PATH environment variable in raynet/build_paths.sh"
    echo ""
    echo "An environment can be created automatically with:"
    echo "    ./build.sh -i"
    echo ""
    echo "Build failed."
    exit 1
fi



build_paths_hook='source "'$RAYNET_PATH'/build_paths.sh"'
activate_script="$RAYNET_VENV_PATH/bin/activate"
if ! grep -Fxq "$build_paths_hook" "$activate_script"; then
    {
        echo ""
        echo "# RayNet environment"
        echo "$build_paths_hook"
    } >> "$activate_script"
fi
source "$RAYNET_VENV_PATH/bin/activate"
echo "Python virtual environment activated. Proceeding with build..."
echo "----------------------------------------------------------------------------------------------------------------"

python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "$python_version" != "3.12" ]; then
    echo ""
    echo "WARNING: Active virtual environment is using Python $python_version"
    echo "RayNet is officially tested with Python 3.12."
    echo ""
    echo "Using another Python version may cause:"
    echo "  - pybind11 ABI incompatibilities"
    echo "  - import failures"
    echo "  - runtime crashes"
    echo ""

    read -p "Do you wish to continue anyway? (y/n): " confirm

    case $confirm in
        [Yy]|[Yy][Ee][Ss])
            echo "Continuing build..."
            ;;
        *)
            echo "Build cancelled."
            exit 1
            ;;
    esac
fi
echo ""

# Command/flags have been validated. Begin building here.
# ----------------------------------------------------------------------------------------------------------------------------------

source $OMNET_PATH/setenv

# List of simlibs. Any simlib that needs to be compiled FIRST (eg. is a dependency) should be added here, rather than the loop.
simlibs=("$RAYNET_PATH/simlibs/RLComponents"
"$RAYNET_PATH/simlibs/tcpPacedNoCC"
)

# list directories in the form "/tmp/dirname/"
for dir in $RAYNET_PATH/simlibs/*/     
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
cd "$INET_PATH" && make -j32 MODE=$mode
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
    cd $RAYNET_PATH
    rm -r build 
fi

# Build raynet (release or debug)
cd $RAYNET_PATH
mkdir build 
cd build
echo "Building RayNet..."
echo "---------------------------------"
cmake -DCMAKE_BUILD_TYPE=$mode -DPython3_EXECUTABLE="$RAYNET_VENV_PATH/bin/python" ../ && \
make -j32
echo "---------------------------------"