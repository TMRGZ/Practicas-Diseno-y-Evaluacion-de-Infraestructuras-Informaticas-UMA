################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH       ?= /usr/local/cuda

OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
OSLOWER = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
	XCODE_GE_5 = $(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5)
endif

# Take command line flags that override any of these settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif
ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif
ifeq ($(ARMv7),1)
	OS_SIZE = 32
	OS_ARCH = armv7l
endif

# Common binaries
ifneq ($(DARWIN),)
ifeq ($(XCODE_GE_5),1)
  GCC ?= clang
else
  GCC ?= g++
endif
else
  GCC ?= g++
endif
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)

# internal flags
NVCCFLAGS   := -m${OS_SIZE}
CCFLAGS     :=
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

# OS-specific build flags
ifneq ($(DARWIN),)
  LDFLAGS += -rpath $(CUDA_PATH)/lib
  CCFLAGS += -arch $(OS_ARCH)
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),androideabi)
      NVCCFLAGS += -target-os-variant Android
    else
      ifeq ($(abi),gnueabi)
        CCFLAGS += -mfloat-abi=softfp
      else
        # default to gnueabihf
        override abi := gnueabihf
        LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
        CCFLAGS += -mfloat-abi=hard
      endif
    endif
  endif
endif

ifeq ($(ARMv7),1)
NVCCFLAGS += -target-cpu-arch ARM
ifneq ($(TARGET_FS),)
CCFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../../common/inc
LIBRARIES :=

################################################################################

UBUNTU = $(shell lsb_release -i -s 2>/dev/null | grep -i ubuntu)

EXEC   ?=

PTX_FILE := matrixmul_kernel${OS_SIZE}.ptx

ifneq ($(DARWIN),)
  ALL_LDFLAGS += -Xlinker -framework -Xlinker CUDA
else
  CUDA_SEARCH_PATH ?=
  ifeq ($(ARMv7),1)
    ifneq ($(TARGET_FS),)
      LIBRARIES += -L$(TARGET_FS)/usr/lib
      CUDA_SEARCH_PATH += $(TARGET_FS)/usr/lib
    endif
    CUDA_SEARCH_PATH += /usr/arm-linux-gnueabihf/lib
  else
    ifneq ($(UBUNTU),)
      CUDA_SEARCH_PATH += /usr/lib
    else
      ifeq ($(OS_SIZE),32)
        CUDA_SEARCH_PATH += /usr/lib
      else
        CUDA_SEARCH_PATH += /usr/lib64
      endif
    endif
  endif

  CUDALIB ?= $(shell find $(CUDA_SEARCH_PATH) -maxdepth 1 -name libcuda.so)
  ifeq ("$(CUDALIB)","")
    $(info >>> WARNING - libcuda.so not found, CUDA Driver is not installed.  Please re-install the driver. <<<)
    EXEC=@echo "[@]"
  endif

  LIBRARIES += -lcuda
endif

################################################################################

# Target rules
all: build

build: matrixmul $(PTX_FILE)

$(PTX_FILE): matrixmul_kernel.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -ptx $<
	$(EXEC) mkdir -p data
	$(EXEC) cp -f $@ ./data
	$(EXEC) mkdir -p ../../bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))
	$(EXEC) cp -f $@ ../../bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))

matrixmul.o: matrixmul.cu matrixmul_gold.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

matrixmul: matrixmul.o matrixmul_gold.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../../bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))
	$(EXEC) cp $@ ../../bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))

run: build
	$(EXEC) ./matrixmul

clean:
	rm -f matrixmul matrixmul.o  data/$(PTX_FILE) $(PTX_FILE)
	rm -rf ../../bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))/matrixmul
	rm -rf ../../bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))/$(PTX_FILE)

clobber: clean
