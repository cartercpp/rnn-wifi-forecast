#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <print>
#include <string>
#include <string_view>
#include <thread>
#include <stop_token>
#include "math_vector.h"
#include "recurrent_neural_network.h"

namespace
{
    constexpr std::string_view RESET   = "\033[0m";
    constexpr std::string_view DIM     = "\033[2m";
    constexpr std::string_view BOLD    = "\033[1m";
    constexpr std::string_view CYAN    = "\033[96m";
    constexpr std::string_view MAGENTA = "\033[95m";
    constexpr std::string_view GREEN   = "\033[92m";
    constexpr std::string_view YELLOW  = "\033[93m";
    constexpr std::string_view WHITE   = "\033[97m";

    struct sample
    {
        double actual{};
        double predicted{};
        bool hasPrediction{};
    };

    std::string sparkline(const std::deque<double>& values)
    {
        static constexpr std::array<std::string_view, 8> bars{
            "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"
        };

        if (values.empty())
            return {};

        const double maxValue = std::max(1.0, *std::max_element(values.begin(), values.end()));
        std::string result;

        for (double value : values)
        {
            const double normalized = std::clamp(value / maxValue, 0.0, 1.0);
            const std::size_t index = static_cast<std::size_t>(std::round(normalized * 7.0));
            result += bars[index];
        }

        return result;
    }

    void render(const std::deque<sample>& history,
                const std::deque<double>& inputHistory,
                std::size_t contextLength,
                std::size_t trainingSteps)
    {
        constexpr std::size_t graphWidth = 54;
        constexpr std::size_t graphHeight = 17;

        double currentActual = history.empty() ? 0.0 : history.back().actual;
        double currentPrediction = 0.0;
        bool hasPrediction = !history.empty() && history.back().hasPrediction;

        if (hasPrediction)
            currentPrediction = history.back().predicted;

        double graphMax = 5.0;
        for (const auto& point : history)
        {
            graphMax = std::max(graphMax, point.actual);
            if (point.hasPrediction)
                graphMax = std::max(graphMax, point.predicted);
        }
        graphMax *= 1.10;

        double rollingMae = 0.0;
        std::size_t predictionCount = 0;
        for (const auto& point : history)
        {
            if (!point.hasPrediction)
                continue;

            rollingMae += std::abs(point.predicted - point.actual);
            ++predictionCount;
        }
        if (predictionCount > 0)
            rollingMae /= static_cast<double>(predictionCount);

        // 0 = blank, 1 = actual traffic area, 2 = prediction marker, 3 = overlap
        std::array<std::array<unsigned char, graphWidth>, graphHeight> grid{};

        const std::size_t visible = std::min(graphWidth, history.size());
        const std::size_t start = history.size() - visible;
        const std::size_t xOffset = graphWidth - visible;

        auto yFor = [graphMax](double value) {
            const double normalized = std::clamp(value / graphMax, 0.0, 1.0);
            return static_cast<std::size_t>(
                std::round((1.0 - normalized) * static_cast<double>(graphHeight - 1))
            );
        };

        for (std::size_t i = 0; i < visible; ++i)
        {
            const auto& point = history[start + i];
            const std::size_t x = xOffset + i;
            const std::size_t actualY = yFor(point.actual);

            for (std::size_t y = actualY; y < graphHeight; ++y)
                grid[y][x] = 1;

            if (point.hasPrediction)
            {
                const std::size_t predictedY = yFor(point.predicted);
                grid[predictedY][x] = (grid[predictedY][x] == 1) ? 3 : 2;
            }
        }

        std::print("\033[H");
        std::print("{}{}╔════════════════════════════════════════════════════════════╗{}\n", BOLD, CYAN, RESET);
        std::print("{}{}║  LIVE NETWORK RNN  //  1-SECOND TRAFFIC FORECAST          ║{}\n", BOLD, CYAN, RESET);
        std::print("{}{}╚════════════════════════════════════════════════════════════╝{}\n", BOLD, CYAN, RESET);
        std::print("{}  INPUT  {:>2}s   HIDDEN  {:>2}   TRAINING  BPTT / ONLINE{}\n\n",
                   DIM, contextLength, contextLength, RESET);

        std::print("  {}ACTUAL     {:>8.2f} Mbps{}\n", CYAN, currentActual, RESET);
        if (hasPrediction)
        {
            const double error = std::abs(currentPrediction - currentActual);
            std::print("  {}PREDICTED  {:>8.2f} Mbps{}\n", MAGENTA, currentPrediction, RESET);
            std::print("  {}ERROR      {:>8.2f} Mbps{}     rolling MAE {:>6.2f}\n",
                       YELLOW, error, RESET, rollingMae);
        }
        else
        {
            std::print("  {}PREDICTED     --.-- Mbps{}\n", MAGENTA, RESET);
            std::print("  {}COLLECTING CONTEXT  {:>2}/{:<2}{}\n",
                       YELLOW, inputHistory.size(), contextLength, RESET);
        }

        std::print("\n  {}recent input  {}{}{}\n\n", DIM, GREEN, sparkline(inputHistory), RESET);

        for (std::size_t y = 0; y < graphHeight; ++y)
        {
            if (y == 0)
                std::print(" {:>6.1f} ┤", graphMax);
            else if (y == graphHeight - 1)
                std::print(" {:>6.1f} ┤", 0.0);
            else
                std::print("        │");

            for (std::size_t x = 0; x < graphWidth; ++x)
            {
                switch (grid[y][x])
                {
                    case 1: std::print("{}█{}", CYAN, RESET); break;
                    case 2: std::print("{}●{}", MAGENTA, RESET); break;
                    case 3: std::print("{}◆{}", WHITE, RESET); break;
                    default: std::print(" "); break;
                }
            }
            std::print("\n");
        }

        std::print("        └");
        for (std::size_t i = 0; i < graphWidth; ++i)
            std::print("─");
        std::print("\n");
        std::print("         {}█ actual traffic{}   {}● RNN prediction{}   {}◆ overlap{}\n",
                   CYAN, RESET, MAGENTA, RESET, WHITE, RESET);
        std::print("\n  {}TRAINING STEPS {:>5}{}   {}● learning live{}\n",
                   BOLD, trainingSteps, RESET, GREEN, RESET);
        std::print("  {}Generate some traffic and watch the forecast adapt.  [ENTER = stop]{}\n", DIM, RESET);
        std::print("\033[J");
    }
}

int main()
{
    constexpr std::string_view rxPath = "/sys/class/net/wlp0s20f3/statistics/rx_bytes";

    auto getRxBytes = [&]() {
        std::ifstream file{rxPath.data()};
        std::uint64_t bytes{};
        file >> bytes;
        return bytes;
    };

    constexpr double minMbps = 0.0;
    constexpr double maxMbps = 1000.0;
    constexpr std::size_t contextLength = 15;
    constexpr std::size_t displayLength = 54;

    std::deque<double> mbpsHistory;
    std::deque<sample> displayHistory;

    recurrent_neural_network trafficNN(contextLength, 0.5); // 0.5 = learning rate

    std::uint64_t prevBytes = getRxBytes();
    auto startInterval = std::chrono::steady_clock::now();
    std::size_t trainingSteps = 0;

    std::print("\033[2J\033[H\033[?25l");

    std::jthread worker{[&](std::stop_token stopToken) {
        while (!stopToken.stop_requested())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            const std::uint64_t bytes = getRxBytes();
            const auto endInterval = std::chrono::steady_clock::now();

            const double changeInMegabits = static_cast<double>(bytes - prevBytes) * (8.0 / 1'000'000.0);
            const double changeInSeconds = std::chrono::duration<double>(endInterval - startInterval).count();
            const double mbps = (changeInSeconds > 0.0) ? changeInMegabits / changeInSeconds : 0.0;
            const double normMbps = std::clamp((mbps - minMbps) / (maxMbps - minMbps), 0.0, 1.0);

            sample newSample{};
            newSample.actual = mbps;

            if (mbpsHistory.size() == contextLength)
            {
                math_vector<double> normHistory(contextLength, 0.0);
                for (std::size_t i = 0; i < contextLength; ++i)
                    normHistory[i] = std::clamp((mbpsHistory[i] - minMbps) / (maxMbps - minMbps), 0.0, 1.0);

                newSample.predicted = trafficNN.predict(normHistory) * (maxMbps - minMbps) + minMbps;
                newSample.hasPrediction = true;

                trafficNN.fit(normHistory, normMbps);
                ++trainingSteps;
                mbpsHistory.pop_front();
            }

            mbpsHistory.push_back(mbps);
            displayHistory.push_back(newSample);
            if (displayHistory.size() > displayLength)
                displayHistory.pop_front();

            render(displayHistory, mbpsHistory, contextLength, trainingSteps);

            prevBytes = bytes;
            startInterval = endInterval;
        }
    }};

    std::cin.get();
    worker.request_stop();
    worker.join();

    std::print("\033[?25h\033[0m\n");
}
