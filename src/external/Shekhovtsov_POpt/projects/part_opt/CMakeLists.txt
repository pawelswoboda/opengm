# Author: shekhovtsov@icg.tugraz.at (Alexander Shekhovtsov)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (part_opt)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

set (VERSION_MAJOR 1)
set (VERSION_MINOR 2)

set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../lib)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../lib)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../bin)
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/../../bin)
SET (SRC ../../code)

SET(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo;Debug" CACHE STRING "RelWithDebInfo;Debug")
set (BUILD_SHARED_LIBS OFF)

#set(CMAKE_CXX_FLAGS "-w -fpermissive -fPIC -fopenmp")

#configure_file (
#  "cfg.h.in"
#  "${PROJECT_BINARY_DIR}/cfg.h"
#  )

# C++11 support.
include(cmake_cpp11/EnableCPP11.cmake)

# Multithreading using OpenMP
# #OPTION(WITH_OPENMP "Enable multi-threading (requires OpenMP)" ON)
SET(WITH_OPENMP ON)
OPTION(WITH_OPENGM "Include test of OpenGM interface" OFF)
# OPTION(MAXFLOW_OPENGM "Use maxflow patch from OpenGM" OFF)
SET(MAXFLOW_OPENGM OFF)

# Library dependencies
SET (LIBRARY_DEPENDENCIES)

#--------------------------------------------------------------
# OPENGM
#--------------------------------------------------------------
if(WITH_OPENGM)
#   message(STATUS "build with OPENGM support")
#   find_package(opengm REQUIRED CONFIG)
    SET(OPENGM_INCLUDE_DIR "Z:/work/opengm/include" CACHE STRING "Z:/work/opengm/include")
    include_directories(${OPENGM_INCLUDE_DIR})
    add_definitions(-DWITH_OPENGM)
endif()

#--------------------------------------------------------------
# HDF5
#--------------------------------------------------------------
if(WITH_OPENGM)
   message(STATUS "build with HDF5 support")
   if(WIN32)
     # FindHDF5 is broken on Windows
     # it wrongly discovers the dlls instead of the lib files
     # see: www.cmake.org/Bug/bug_relationship_graph.php?bug_id=14111
     # therefore we set the variable manually
     find_library( HDF5_IMPORT_LIB NAMES hdf5 hdf5dll )
     find_path(HDF5_INCLUDE_DIR hdf5.h)
     set(HDF5_LIBRARIES ${HDF5_IMPORT_LIB})
     message(STATUS "HDF5: autodiscovery is broken on WIN32; using global search paths")
     message(STATUS "HDF5: found ${HDF5_LIBRARIES}")
     message(STATUS "HDF5_LIBRARIES:" ${HDF5_LIBRARIES})
   else()
     IF(APPLE)
	find_library(HDF5_IMPORT_LIB NAMES hdf5)
	find_path(HDF5_INCLUDE_DIR hdf5.h)
	set(HDF5_LIBRARIES ${HDF5_IMPORT_LIB})
     ELSE()	
	find_path(HDF5_INCLUDE_DIR hdf5.h)
	message(STATUS "HDF5_INCLUDE_DIR: ${HDF5_INCLUDE_DIR}")
	find_package(HDF5 REQUIRED)
     ENDIF()
	message(STATUS "HDF5_LIBRARIES:" ${HDF5_LIBRARIES})
   endif()
   
   get_filename_component(HDF_LIB_DIR ${HDF5_LIBRARIES} PATH)
   message(STATUS "HDF_LIB_DIR:" ${HDF_LIB_DIR})
   LINK_DIRECTORIES(${HDF_LIB_DIR})

if(WIN32)
     find_library( HDF5_HL_IMPORT_LIB NAMES hdf5_hl hdf5_hldll )
     LIST(APPEND HDF5_LIBRARIES ${HDF5_HL_IMPORT_LIB} )
     message(STATUS "HDF5_LIBRARIES:" ${HDF5_LIBRARIES})
endif()
IF(APPLE)
  SET(HDF5_LIBSZ "libszip.a" CACHE STRING "only lib name: libszip.a")
  SET(HDF5_LIBZ "libz.a" CACHE STRING "only lib name: libz.a")
    LIST(APPEND HDF5_LIBRARIES ${HDF_LIB_DIR}/${HDF5_LIBSZ} ${HDF_LIB_DIR}/${HDF5_LIBZ})
   message(STATUS "HDF5_LIBRARIES:" ${HDF5_LIBRARIES})
ENDIF(APPLE)

   include_directories(${HDF5_INCLUDE_DIR})
   add_definitions(${HDF5_CPPFLAGS})
   add_definitions(-DWITH_HDF5)
   LIST(APPEND LIBRARY_DEPENDENCIES ${HDF5_LIBRARIES})
else()
   message(STATUS "build without HDF5")
endif()

if((WITH_OPENGM) AND (BLA))
    message(STATUS "build with HDF5 support")
    find_package(HDF5 REQUIRED)
    include_directories(${HDF5_INCLUDE_DIR})
    add_definitions(-DWITH_HDF5)
    SET(HDF5_LIB "${HDF5_LIBRARIES}" CACHE STRING "${HDF5_LIBRARIES}")
    message(STATUS "HDF5_LIBRARIES=" ${HDF5_LIBRARIES})
    message(STATUS "HDF5_LIB=" ${HDF5_LIB})
# does not work for me, HDF5_LIBRARIES is wrong and HDF5_LIBRARY_DIRS is empty
# have to hack it manually
IF(APPLE)
  SET(HDF5_LIBSZ "libszip.a" CACHE STRING "only lib name: libszip.a")
  SET(HDF5_LIBZ "libz.a" CACHE STRING "only lib name: libz.a")
    get_filename_component(HDF_LIB_DIR ${HDF5_LIB} PATH)
    message(STATUS "HDF5_LIB_DIR=" ${HDF_LIB_DIR})
    SET(U_HDF_LIBS ${HDF5_LIB})
    LIST(APPEND U_HDF_LIBS ${HDF_LIB_DIR}/${HDF5_LIBSZ} ${HDF_LIB_DIR}/${HDF5_LIBZ})
    message(STATUS "U_HDF_LIBS=" ${U_HDF_LIBS})
ENDIF(APPLE)
else()
   #message(STATUS "build without OPENGM")
endif()
#--------------------------------------------------------------
# MaxFlow
#--------------------------------------------------------------

if((WITH_OPENGM) AND (MAXFLOW_OPENGM))
	message(STATUS "build with external inference algorithm MaxFlow")
	SET(MAXFLOW_PATCHEDSRCDIR "${OPENGM_INCLUDE_DIR}/../src/external/MaxFlow-v3.02.src-patched/" CACHE STRING "MAXFLOW patched source code directory")
	add_definitions(-DWITH_MAXFLOW)
	include_directories(${MAXFLOW_PATCHEDSRCDIR})
endif((WITH_OPENGM) AND (MAXFLOW_OPENGM))

#---------------compiler setings-------------------------------------------
#add_definitions(-DNO_MEM_MANAGER)

## For Clang-based tools.
##SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

IF (WITH_OPENMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    MESSAGE("-- Found OpenMP.")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ADD_DEFINITIONS(-DUSE_OPENMP)
  ELSE(OPENMP_FOUND)
    MESSAGE("-- Can't find OpenMP. Continuing without it.")
  ENDIF(OPENMP_FOUND)
ELSE (WITH_OPENMP)
  MESSAGE("-- OpenMP is disabled. Continuing without it.")
ENDIF (WITH_OPENMP)

IF (OPENMP_FOUND)
  IF (NOT MSVC)
    LIST(APPEND LIBRARY_DEPENDENCIES gomp)
  ENDIF (NOT MSVC)
ENDIF (OPENMP_FOUND)

# LIST(APPEND LIBRARY_DEPENDENCIES maxflow3)


#
# MSVC settings
#
IF (MSVC)
  # Disable deprecation warning for standard functions.
  ADD_DEFINITIONS("/wd4996")
  # To get rid of annoying min and max macros if windows.h
  # is included.
  ADD_DEFINITIONS("-DNOMINMAX=1")
  # Macro redefinitions.
  ADD_DEFINITIONS("/wd4005")
ELSE()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -w -fpermissive -O3")
  ###SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
 
ENDIF (MSVC)

IF(CMAKE_COMPILER_IS_GNUCC)
  LIST(APPEND LIBRARY_DEPENDENCIES rt)
ENDIF()


include_directories("${PROJECT_BINARY_DIR}")
include_directories(${SRC})

# function to find header files
function(find_headers out_header_list_name) # example: find_headers(HH ${SRCLIST})
	LIST(REMOVE_AT ARGV 0) # remove the first argument = out_header_list_name
	SET(HH)
	FOREACH(file ${ARGV})
		#message(${file})
		get_filename_component(ffile ${file} REALPATH) # full absolute path
		get_filename_component(dir ${ffile} PATH) # just path
		#message(${dir})
		set(H)
		set(HXX)
		set(HPP)
		file(GLOB_RECURSE H "${dir}/*.h")
		file(GLOB_RECURSE HXX "${dir}/*.hxx")
		file(GLOB_RECURSE HPP "${dir}/*.hpp")
		LIST(APPEND HH ${H} ${HXX} ${HPP})
	ENDFOREACH()
	LIST(REMOVE_DUPLICATES HH)
	#message(${HH})
	set(${out_header_list_name} ${HH} PARENT_SCOPE)
endfunction(find_headers)

# function to generate nice project files for IDEs
function(source_group_path srcdir)
get_filename_component(SRCDIR ${srcdir} REALPATH) # make sure relative pathes does not get confused
	LIST(REMOVE_AT ARGV 0) # remove the first argument = srcdir
	FOREACH(file ${ARGV})
		get_filename_component(ffile ${file} REALPATH)
		FILE(RELATIVE_PATH pth ${SRCDIR} ${ffile})
		get_filename_component(rdir ${pth} PATH)
		string(REPLACE "/" "\\" rrdir ${rdir})
		get_filename_component(name ${pth} NAME)
        	#MESSAGE(${rrdir} : ${name})
		source_group(${rrdir} FILES ${file})
		#ALLILUJA!
	ENDFOREACH()
endfunction(source_group_path)



message(STATUS "CMAKE_CXX_FLAGS_RELEASE=" ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "CMAKE_EXE_LINKER_FLAGS=" ${CMAKE_EXE_LINKER_FLAGS})
message(STATUS "LIBRARY_DEPENDENCIES=" ${LIBRARY_DEPENDENCIES})

#____________exttype lib________________ 
SET(CPP)
list(APPEND CPP 
  ${SRC}/dynamic/block_allocator.cpp
  ${SRC}/data/dataset.cpp
  ${SRC}/exttype/pvect.cpp
  ${SRC}/geom/vectn.cpp
  ${SRC}/files/xfs.cpp
  ${SRC}/debug/logs.cpp
  ${SRC}/debug/performance.cpp
  ${SRC}/streams/xstringstream.cpp
  ${SRC}/streams/file_stream.cpp
  ${SRC}/streams/binary_stream.cpp
  ${SRC}/streams/text_stream.cpp
)
find_headers(HH ${CPP})
source_group_path(${SRC} ${CPP})
source_group_path(${SRC} ${HH})
add_library(exttype ${CPP} ${HH})


#__________maxflow lib__________________ 
SET(CPP)
IF((WITH_OPENGM) AND (MAXFLOW_OPENGM))
	list(APPEND CPP ${MAXFLOW_PATCHEDSRCDIR}/graph.cpp ${MAXFLOW_PATCHEDSRCDIR}/maxflow.cpp)
ELSE((WITH_OPENGM) AND (MAXFLOW_OPENGM))
	list(APPEND CPP ${SRC}/maxflow/graph.cpp ${SRC}/maxflow/maxflow.cpp)
ENDIF((WITH_OPENGM) AND (MAXFLOW_OPENGM))

find_headers(HH ${CPP})
source_group_path(${SRC} ${CPP})
source_group_path(${SRC} ${HH})
add_library(maxflow_BK ${CPP} ${HH})

#___________part_opt lib________________
SET(CPP)
list(APPEND CPP 
${SRC}/optim/graph/mgraph.cpp
${SRC}/optim/trws/stream_graph.cpp
${SRC}/optim/part_opt/dee.cpp
${SRC}/optim/part_opt/dee2.cpp
${SRC}/optim/part_opt/energy.cpp
${SRC}/optim/part_opt/part_opt_TRWS.cpp
${SRC}/optim/part_opt/part_opt_interface.cpp
${SRC}/optim/part_opt/energy.cpp
${SRC}/optim/part_opt/msg_alg.cpp
${SRC}/optim/part_opt/trws_machine.cpp
${SRC}/optim/part_opt/vectorizers.cpp
)

find_headers(HH ${CPP})
source_group_path(${SRC} ${CPP})
source_group_path(${SRC} ${HH})
add_library(part_opt ${CPP} ${HH})

#______________tests_____________________

#______________random____________________
SET(CPP)
list(APPEND CPP 
${SRC}/optim/part_opt/test_random.cpp
)
source_group("" FILES ${CPP})
add_executable(test_random ${CPP})
target_link_libraries (test_random part_opt exttype maxflow_BK ${LIBRARY_DEPENDENCIES})
#______________opengm____________________
if(WITH_OPENGM)
SET(CPP)
list(APPEND CPP
${SRC}/optim/part_opt/test_part_opt_opengm.cpp
)
source_group("" FILES ${CPP})
add_executable(test_part_opt_opengm ${CPP})
target_link_libraries (test_part_opt_opengm part_opt exttype maxflow_BK ${LIBRARY_DEPENDENCIES})
ENDIF(WITH_OPENGM)

