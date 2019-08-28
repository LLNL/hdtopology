############## Always using highest warning level ###############
if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # Update if necessary
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Wno-long-long -Wno-reorder -Wno-format -Wno-unknown-pragmas -Wno-sign-compare") #-pedantic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  #-Wno-sign-compare topo code
endif()
#################################################################

include_directories(
   ./
   #./external/
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


# Now we define internal variables controlling the installation
IF (${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "Install directory" FORCE)
ENDIF()

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(utilities)
#ADD_SUBDIRECTORY(external)

IF (ENABLE_PYTHON)
    ADD_SUBDIRECTORY(python)
ENDIF()
