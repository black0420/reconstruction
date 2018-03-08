# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fubo/reconstruction/dense_monocular

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fubo/reconstruction/dense_monocular/build

# Include any dependencies generated for this target.
include CMakeFiles/dense_mapping.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dense_mapping.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dense_mapping.dir/flags.make

CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o: CMakeFiles/dense_mapping.dir/flags.make
CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o: ../dense_mapping.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/fubo/reconstruction/dense_monocular/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o -c /home/fubo/reconstruction/dense_monocular/dense_mapping.cpp

CMakeFiles/dense_mapping.dir/dense_mapping.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dense_mapping.dir/dense_mapping.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/fubo/reconstruction/dense_monocular/dense_mapping.cpp > CMakeFiles/dense_mapping.dir/dense_mapping.cpp.i

CMakeFiles/dense_mapping.dir/dense_mapping.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dense_mapping.dir/dense_mapping.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/fubo/reconstruction/dense_monocular/dense_mapping.cpp -o CMakeFiles/dense_mapping.dir/dense_mapping.cpp.s

CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o.requires:
.PHONY : CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o.requires

CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o.provides: CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o.requires
	$(MAKE) -f CMakeFiles/dense_mapping.dir/build.make CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o.provides.build
.PHONY : CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o.provides

CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o.provides.build: CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o

# Object files for target dense_mapping
dense_mapping_OBJECTS = \
"CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o"

# External object files for target dense_mapping
dense_mapping_EXTERNAL_OBJECTS =

dense_mapping: CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o
dense_mapping: CMakeFiles/dense_mapping.dir/build.make
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_calib3d.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_core.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_features2d.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_flann.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_highgui.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_imgcodecs.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_imgproc.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_ml.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_objdetect.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_photo.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_shape.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_stitching.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_superres.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_video.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_videoio.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_videostab.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_viz.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_phase_unwrapping.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_rgbd.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_saliency.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_xfeatures2d.so.3.2.0
dense_mapping: /home/fubo/Sophus/build/libSophus.so
dense_mapping: /home/fubo/Pangolin/build/src/libpangolin.so
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_objdetect.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_shape.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_photo.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_video.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_calib3d.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_features2d.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_flann.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_highgui.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_ml.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_videoio.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_imgcodecs.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_imgproc.so.3.2.0
dense_mapping: /home/fubo/maplab_ws/devel/lib/libopencv_core.so.3.2.0
dense_mapping: /usr/lib/x86_64-linux-gnu/libGLU.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libGL.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libSM.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libICE.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libX11.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libXext.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libGLEW.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libSM.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libICE.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libX11.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libXext.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libGLEW.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libpython2.7.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libdc1394.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libavcodec.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libavformat.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libavutil.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libswscale.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libpng.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libz.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libjpeg.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libtiff.so
dense_mapping: /usr/lib/x86_64-linux-gnu/libIlmImf.so
dense_mapping: CMakeFiles/dense_mapping.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable dense_mapping"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dense_mapping.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dense_mapping.dir/build: dense_mapping
.PHONY : CMakeFiles/dense_mapping.dir/build

CMakeFiles/dense_mapping.dir/requires: CMakeFiles/dense_mapping.dir/dense_mapping.cpp.o.requires
.PHONY : CMakeFiles/dense_mapping.dir/requires

CMakeFiles/dense_mapping.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dense_mapping.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dense_mapping.dir/clean

CMakeFiles/dense_mapping.dir/depend:
	cd /home/fubo/reconstruction/dense_monocular/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fubo/reconstruction/dense_monocular /home/fubo/reconstruction/dense_monocular /home/fubo/reconstruction/dense_monocular/build /home/fubo/reconstruction/dense_monocular/build /home/fubo/reconstruction/dense_monocular/build/CMakeFiles/dense_mapping.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dense_mapping.dir/depend

