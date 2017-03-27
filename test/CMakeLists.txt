file(GLOB TEST_TARGETS test_*.cxx)

FOREACH(FILE ${TEST_TARGETS})
    GET_FILENAME_COMPONENT(TARGET ${FILE} NAME_WE)
    add_executable(${TARGET} ${FILE})
    target_link_libraries(${TARGET} vigra_core)
    add_test(${TARGET} ${TARGET})
ENDFOREACH(FILE)
