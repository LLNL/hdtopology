INCLUDE_DIRECTORIES(
   ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

ADD_EXECUTABLE(unit_test unit_test.cpp)
TARGET_LINK_LIBRARIES(unit_test HDFileFormat)

ADD_EXECUTABLE(unit_test2 unit_test2.cpp)
TARGET_LINK_LIBRARIES(unit_test2 HDFileFormat)

ADD_EXECUTABLE(unit_test3 unit_test3.cpp)
TARGET_LINK_LIBRARIES(unit_test3 HDFileFormat)

ADD_EXECUTABLE(unit_test6 unit_test6.cpp)
TARGET_LINK_LIBRARIES(unit_test6 HDFileFormat)

ADD_EXECUTABLE(unit_test_metaInfo_write unit_test_metaInfo_write.cpp)
TARGET_LINK_LIBRARIES(unit_test_metaInfo_write HDFileFormat)

ADD_EXECUTABLE(unit_test_metaInfo_read unit_test_metaInfo_read.cpp)
TARGET_LINK_LIBRARIES(unit_test_metaInfo_read HDFileFormat)

ADD_EXECUTABLE(unit_test_metaInfo_rewrite unit_test_metaInfo_rewrite.cpp)
TARGET_LINK_LIBRARIES(unit_test_metaInfo_rewrite HDFileFormat)

ADD_EXECUTABLE(unit_test_compression unit_test_compression.cpp)
TARGET_LINK_LIBRARIES(unit_test_compression HDFileFormat)

ADD_EXECUTABLE(cmdDump command_line_info_dump.cpp)
TARGET_LINK_LIBRARIES(cmdDump HDFileFormat)

