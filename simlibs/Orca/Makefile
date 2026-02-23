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
	cd src && opp_makemake -o Orca --make-so -f --deep --mode ${MODE} -D${RAYNET_FEATURE} -I${RAYNET_HOME}/simlibs/RLComponents/src -L${RAYNET_HOME}/simlibs/RLComponents/src -lRLComponents${LIB_SUFFIX} -I${HOME}/inet4.5/src -L${HOME}/inet4.5/src -lINET${LIB_SUFFIX} -I${HOME}/raynet/simlibs/tcpPaced/src -L${HOME}/raynet/simlibs/tcpPaced/src -ltcpPaced${LIB_SUFFIX}  -I${HOME}/raynet/simlibs/cubic/src -L${HOME}/raynet/simlibs/cubic/src -lcubic${LIB_SUFFIX}  

checkmakefiles:
	@if [ ! -f src/Makefile ]; then \
	echo; \
	echo '======================================================================='; \
	echo 'src/Makefile does not exist. Please use "make makefiles" to generate it!'; \
	echo '======================================================================='; \
	echo; \
	exit 1; \
	fi