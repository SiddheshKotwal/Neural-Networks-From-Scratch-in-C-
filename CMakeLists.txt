cmake_minimum_required(VERSION 3.16)

project(NeuralNetworksFromScratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP QUIET)

set(CORE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/Neural-Networks-From-Scratch-in-CPP"
)

set(CORE_EXAMPLES
    binary_logistic_regression
    classification_neural_network
    forward_pass_neural_network
    random_parameters
    regression_model
    testing_datasets
)

foreach(example IN LISTS CORE_EXAMPLES)

    add_executable(
        ${example}
        "${CORE_DIR}/${example}.cpp"
    )

    target_include_directories(
        ${example}
        PRIVATE
        "${CORE_DIR}"
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            ${example}
            PRIVATE
            OpenMP::OpenMP_CXX
        )
    endif()

endforeach()

option(
    BUILD_REAL_DATASET
    "Build Fashion-MNIST examples when OpenCV is available"
    OFF
)

if(BUILD_REAL_DATASET)

    find_package(OpenCV QUIET)

    if(OpenCV_FOUND)

        foreach(example IN ITEMS
            main
            predictions
            loading_and_testing_params
        )

            add_executable(
                real_dataset_${example}
                "${CMAKE_CURRENT_SOURCE_DIR}/A Real Dataset/${example}.cpp"
            )

            target_include_directories(
                real_dataset_${example}
                PRIVATE
                "${CORE_DIR}"
                ${OpenCV_INCLUDE_DIRS}
            )

            target_link_libraries(
                real_dataset_${example}
                PRIVATE
                ${OpenCV_LIBS}
            )

            if(OpenMP_CXX_FOUND)
                target_link_libraries(
                    real_dataset_${example}
                    PRIVATE
                    OpenMP::OpenMP_CXX
                )
            endif()

        endforeach()

    else()

        message(
            WARNING
            "OpenCV not found; real-dataset targets are disabled."
        )

    endif()

endif()
