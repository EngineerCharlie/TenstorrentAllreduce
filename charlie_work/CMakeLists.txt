set(PROGRAMMING_EXAMPLES_SRCS
    # ${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_compute2/add_2_integers_in_compute2.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/all_red_swing_1D/all_red_swing_1D.cpp # <------------------
    ${CMAKE_CURRENT_SOURCE_DIR}/allred_BO_2D/allred_BO_2D.cpp # <------------------
    ${CMAKE_CURRENT_SOURCE_DIR}/allred_LO_2D/allred_LO_2D.cpp # <------------------
    ${CMAKE_CURRENT_SOURCE_DIR}/allred_mem_2D/allred_mem_2D.cpp # <------------------
    # ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer_tile_addition/circular_buffer_tile_addition.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/swing_multicore/swing_multicore.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/swing_multicore_1D/swing_multicore_1D.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/swing_multicore_1D/swing_multicore_1D_y.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/recdub_multicore_1D/recdub_multicore_1D.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/swing_multicore_2D/swing_multicore_2D.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/recdub_multicore_2D/recdub_multicore_2D.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/hello_core2core_1d/hello_core2core_1d.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/hello_core2core_array/hello_core2core_array.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/hello_core2core_arr_simple/hello_core2core_arr_simple.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/hello_core2core_snake/hello_core2core_snake.cpp
)

include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/allred_helper)

CREATE_PGM_EXAMPLES_EXE("${PROGRAMMING_EXAMPLES_SRCS}" "charlie_work")

foreach(EXE_NAME allred_BO_2D allred_LO_2D allred_mem_2D)
    target_sources(${EXE_NAME}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allred_helper/allred_helper.cpp
    )
endforeach()
