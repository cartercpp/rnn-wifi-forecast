# RNN Wi-Fi Traffic Forecast

A live terminal visualization that uses a recurrent neural network written from scratch in C++ to predict incoming Wi-Fi traffic one second into the future.

The program reads network receive statistics directly from Linux, trains the RNN online as new samples arrive, and displays the actual traffic, prediction, error, rolling MAE, and recent traffic history in real time.

## How it works

1. Read received bytes from `/sys/class/net/<interface>/statistics/rx_bytes` once per second.
2. Convert the change in bytes into Mbps.
3. Keep the most recent 15 traffic samples as the RNN input sequence.
4. Predict the next normalized traffic value.
5. Train on the newly observed value using backpropagation through time (BPTT).
6. Render the actual and predicted traffic in the terminal.

The neural network implementation is built from scratch using the custom `math_vector` and `matrix` classes in this repository. No machine-learning library is used.

## Visualization

The terminal display includes:

- live incoming traffic in Mbps
- the RNN's one-second-ahead prediction
- current prediction error
- rolling mean absolute error
- a sparkline of recent input samples
- a scrolling graph of actual traffic and predictions
- the number of online training steps completed

Generate network traffic while the program is running to watch the model adapt in real time.

## RNN architecture

The network is a small recurrent neural network with:

- one scalar input per timestep
- 15 hidden units
- tanh hidden activations
- a sigmoid output
- recurrent hidden-to-hidden weights
- online gradient descent
- backpropagation through time across the 15-sample sequence

## Requirements

- Linux
- CMake 4.3+
- a compiler with C++26 support
- a terminal with ANSI escape-sequence and Unicode support

The network interface is currently set in `main.cpp` as:

```cpp
constexpr std::string_view rxPath = "/sys/class/net/wlp0s20f3/statistics/rx_bytes";
```

If your Wi-Fi interface has a different name, replace `wlp0s20f3` with your interface name. You can find available interfaces with:

```bash
ls /sys/class/net
```

## Build and run

```bash
git clone https://github.com/cartercpp/rnn-wifi-forecast.git
cd rnn-wifi-forecast
cmake -S . -B build
cmake --build build
./build/RNNTrafficForecasting
```

Press **Enter** to stop the program.

## Files

- `main.cpp` — network sampling, online training, and terminal visualization
- `recurrent_neural_network.h` — RNN forward pass and BPTT training
- `math_vector.h` — vector operations used by the network
- `matrix.h` — matrix operations used by the recurrent layer

## Notes

This is an educational from-scratch implementation intended to make recurrent neural networks and online time-series prediction visually understandable. It is not intended to be a production network-forecasting system.
