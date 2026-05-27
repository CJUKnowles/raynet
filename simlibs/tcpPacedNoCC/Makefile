MODE ?= release
ifeq ($(MODE),debug)
    LIB_SUFFIX = _dbg # Append _dbg to certain libraries if building in debug mode.
else
    LIB_SUFFIX = 
endif

all: checkmakefiles
	cd src && $(MAKE)

clean: checkmakefiles
	cd src && $(MAKE) clean

cleanall: checkmakefiles
	cd src && $(MAKE) MODE=release clean
	cd src && $(MAKE) MODE=debug clean
	rm -f src/Makefile

makefiles:
	cd src && opp_makemake -o tcpPacedNoCC --make-so -f --deep --mode ${MODE} -I${INET_PATH}/src -L${INET_PATH}/src -lINET${LIB_SUFFIX} -I${OMNET_PATH}/samples/tcpPaced/src -L${OMNET_PATH}/samples/tcpPaced/src -ltcpPaced${LIB_SUFFIX}  -I${OMNET_PATH}/samples/cubic/src -L${OMNET_PATH}/samples/cubic/src -lcubic${LIB_SUFFIX}   

checkmakefiles:
	@if [ ! -f src/Makefile ]; then \
	echo; \
	echo '======================================================================='; \
	echo 'src/Makefile does not exist. Please use "make makefiles" to generate it!'; \
	echo '======================================================================='; \
	echo; \
	exit 1; \
	fi