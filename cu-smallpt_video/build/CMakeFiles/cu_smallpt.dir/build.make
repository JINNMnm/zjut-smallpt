# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /ssd10/jxc/smallpt/cu-smallpt_video

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /ssd10/jxc/smallpt/cu-smallpt_video/build

# Include any dependencies generated for this target.
include CMakeFiles/cu_smallpt.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cu_smallpt.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cu_smallpt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cu_smallpt.dir/flags.make

CMakeFiles/cu_smallpt.dir/src/kernel.cu.o: CMakeFiles/cu_smallpt.dir/flags.make
CMakeFiles/cu_smallpt.dir/src/kernel.cu.o: ../src/kernel.cu
CMakeFiles/cu_smallpt.dir/src/kernel.cu.o: CMakeFiles/cu_smallpt.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/ssd10/jxc/smallpt/cu-smallpt_video/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cu_smallpt.dir/src/kernel.cu.o"
	/usr/local/cuda-12.1/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cu_smallpt.dir/src/kernel.cu.o -MF CMakeFiles/cu_smallpt.dir/src/kernel.cu.o.d -x cu -dc /ssd10/jxc/smallpt/cu-smallpt_video/src/kernel.cu -o CMakeFiles/cu_smallpt.dir/src/kernel.cu.o

CMakeFiles/cu_smallpt.dir/src/kernel.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cu_smallpt.dir/src/kernel.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cu_smallpt.dir/src/kernel.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cu_smallpt.dir/src/kernel.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cu_smallpt
cu_smallpt_OBJECTS = \
"CMakeFiles/cu_smallpt.dir/src/kernel.cu.o"

# External object files for target cu_smallpt
cu_smallpt_EXTERNAL_OBJECTS =

CMakeFiles/cu_smallpt.dir/cmake_device_link.o: CMakeFiles/cu_smallpt.dir/src/kernel.cu.o
CMakeFiles/cu_smallpt.dir/cmake_device_link.o: CMakeFiles/cu_smallpt.dir/build.make
CMakeFiles/cu_smallpt.dir/cmake_device_link.o: CMakeFiles/cu_smallpt.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/ssd10/jxc/smallpt/cu-smallpt_video/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/cu_smallpt.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cu_smallpt.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cu_smallpt.dir/build: CMakeFiles/cu_smallpt.dir/cmake_device_link.o
.PHONY : CMakeFiles/cu_smallpt.dir/build

# Object files for target cu_smallpt
cu_smallpt_OBJECTS = \
"CMakeFiles/cu_smallpt.dir/src/kernel.cu.o"

# External object files for target cu_smallpt
cu_smallpt_EXTERNAL_OBJECTS =

cu_smallpt: CMakeFiles/cu_smallpt.dir/src/kernel.cu.o
cu_smallpt: CMakeFiles/cu_smallpt.dir/build.make
cu_smallpt: CMakeFiles/cu_smallpt.dir/cmake_device_link.o
cu_smallpt: CMakeFiles/cu_smallpt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/ssd10/jxc/smallpt/cu-smallpt_video/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable cu_smallpt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cu_smallpt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cu_smallpt.dir/build: cu_smallpt
.PHONY : CMakeFiles/cu_smallpt.dir/build

CMakeFiles/cu_smallpt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cu_smallpt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cu_smallpt.dir/clean

CMakeFiles/cu_smallpt.dir/depend:
	cd /ssd10/jxc/smallpt/cu-smallpt_video/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /ssd10/jxc/smallpt/cu-smallpt_video /ssd10/jxc/smallpt/cu-smallpt_video /ssd10/jxc/smallpt/cu-smallpt_video/build /ssd10/jxc/smallpt/cu-smallpt_video/build /ssd10/jxc/smallpt/cu-smallpt_video/build/CMakeFiles/cu_smallpt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cu_smallpt.dir/depend

