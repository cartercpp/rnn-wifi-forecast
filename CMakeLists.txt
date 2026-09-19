cmake_minimum_required(VERSION 4.3)
project(RNNTrafficForecasting)

set(CMAKE_CXX_STANDARD 26)

add_executable(RNNTrafficForecasting main.cpp
        math_vector.h
        matrix.h
        recurrent_neural_network.h)
