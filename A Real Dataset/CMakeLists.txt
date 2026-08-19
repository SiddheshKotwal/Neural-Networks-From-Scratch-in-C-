cmake_minimum_required(VERSION 3.16)

project(RealDatasetExamples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV QUIET)
find_package(OpenMP QUIET)

if(NOT OpenCV_FOUND)

    message(
        WARNING
        "OpenCV not found; real dataset executable targets are not available."
    )

    return()

endif()

set(SOURCES
    main
    predictions
    loading_and_testing_params
)

foreach(example IN LISTS SOURCES)

    add_executable(
        ${example}
        "${example}.cpp"
    )

    target_include_directories(
        ${example}
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../Neural-Networks-From-Scratch-in-CPP"
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(
        ${example}
        PRIVATE
        ${OpenCV_LIBS}
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            ${example}
            PRIVATE
            OpenMP::OpenMP_CXX
        )
    endif()

endforeach()
