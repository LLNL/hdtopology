# - Try to find FLANN
# Once done this will define
#
#  FLANN_FOUND        - system has FLANN
#  FLANN_INCLUDE_DIR  - the FLANN include directory
#  FLANN_LIBRARY      - Link these to use FLANN
#

IF (FLANN_INCLUDE_DIRS)
  # Already in cache, be silent
  SET(FLANN_FIND_QUIETLY TRUE)
ENDIF (FLANN_INCLUDE_DIRS)

FIND_PATH( FLANN_INCLUDE_DIR flann/flann.h
PATHS "external/flann/src/cpp/" "/usr/include" "../" "../flann/src/cpp")

if( WIN32 )

 message(status "FLANN library finder not tested for windows yet!")

else (WIN32)

FIND_LIBRARY( FLANN_LIBRARY
               NAMES flann
           PATHS external/flann/build/lib ../flann/build/lib /lib /usr/lib /usr/lib64 /usr/local/lib)

endif( WIN32)


IF (FLANN_INCLUDE_DIR AND FLANN_LIBRARY)
  SET(FLANN_FOUND TRUE)
ELSE (FLANN_INCLUDE_DIR AND FLANN_LIBRARY)
  SET( FLANN_FOUND FALSE )
ENDIF (FLANN_INCLUDE_DIR AND FLANN_LIBRARY)
