//
// Created by cartercpp on 9/19/26.
//

#ifndef RNNTRAFFICFORECASTING_RECURRENT_NEURAL_NETWORK_H
#define RNNTRAFFICFORECASTING_RECURRENT_NEURAL_NETWORK_H

#include <vector>
#include <random>
#include <cstddef>
#include <cmath>
#include "math_vector.h"
#include "matrix.h"

class recurrent_neural_network
{
public:

    // CONSTRUCTORS

    explicit recurrent_neural_network(std::size_t hiddenSize, double learningRate)
        : m_inputToHidden(hiddenSize, 0), m_hiddenToHidden(hiddenSize, hiddenSize, 0), m_hiddenBias(hiddenSize, 0),
        m_hiddenToOutput(hiddenSize, 0), m_outputBias{0}, m_learningRate{learningRate}, m_hiddenSize{hiddenSize}
    {
        std::random_device rd;
        std::uniform_real_distribution<double> dist(-0.1, 0.1);

        for (std::size_t i = 0; i < hiddenSize; ++i)
        {
            for (std::size_t i2 = 0; i2 < hiddenSize; ++i2)
                m_hiddenToHidden[i][i2] = dist(rd);

            m_inputToHidden[i] = dist(rd);
            m_hiddenToOutput[i] = dist(rd);
        }
    }

    // METHODS

    double predict(const math_vector<double>& input) const
    {
        const std::vector<math_vector<double>> hiddenStates{Forward(input)};
        return Sigmoid(m_hiddenToOutput * hiddenStates.back() + m_outputBias);
    }

    void fit(const math_vector<double>& input, double target)
    {
        const std::vector<math_vector<double>> hiddenStates{Forward(input)};
        const double prediction = Sigmoid(m_hiddenToOutput * hiddenStates.back() + m_outputBias);
        const double delta = prediction - target;

        const double outputGradient = delta * SigmoidDerivative(prediction);
        const math_vector<double> hiddenToOutputGradient = outputGradient * hiddenStates.back();

        math_vector<double> totalInputToHiddenGradient(m_hiddenSize, 0);
        matrix<double> totalHiddenToHiddenGradient(m_hiddenSize, m_hiddenSize, 0);
        math_vector<double> totalHiddenBiasGradient(m_hiddenSize, 0);

        math_vector<double> hiddenStateGradient = outputGradient * m_hiddenToOutput;
        for (std::size_t iter = 0; iter < input.size(); ++iter)
        {
            const std::size_t t = input.size() - iter - 1;
            const auto& hiddenState{hiddenStates[t + 1]};
            const auto& prevHiddenState{hiddenStates[t]};

            hiddenStateGradient = hiddenStateGradient.multiply(TanhDerivative(hiddenState));

            const auto inputToHiddenGradient = hiddenStateGradient * input[t];
            const auto hiddenToHiddenGradient = outer_product(hiddenStateGradient, prevHiddenState);
            const auto& hiddenBiasGradient = hiddenStateGradient;

            totalInputToHiddenGradient += inputToHiddenGradient;
            totalHiddenToHiddenGradient += hiddenToHiddenGradient;
            totalHiddenBiasGradient += hiddenBiasGradient;

            hiddenStateGradient = m_hiddenToHidden.transpose() * hiddenStateGradient;
        }

        m_inputToHidden -= m_learningRate * totalInputToHiddenGradient;
        m_hiddenToHidden -= m_learningRate * totalHiddenToHiddenGradient;
        m_hiddenToOutput -= m_learningRate * hiddenToOutputGradient;
        m_hiddenBias -= m_learningRate * totalHiddenBiasGradient;
        m_outputBias -= m_learningRate * outputGradient;
    }

private:

    static double Sigmoid(double x)
    {
        return 1 / (1 + std::exp(-x));
    }

    static double SigmoidDerivative(double x)
    {
        return x * (1 - x);
    }

    static math_vector<double> Tanh(math_vector<double> vec)
    {
        for (std::size_t i = 0; i < vec.size(); ++i)
            vec[i] = std::tanh(vec[i]);

        return vec;
    }

    static math_vector<double> TanhDerivative(math_vector<double> vec)
    {
        for (std::size_t i = 0; i < vec.size(); ++i)
            vec[i] = 1 - vec[i] * vec[i];

        return vec;
    }

    std::vector<math_vector<double>> Forward(const math_vector<double>& input) const
    {
        std::vector<math_vector<double>> hiddenStates;
        hiddenStates.reserve(input.size() + 1);
        hiddenStates.emplace_back(m_hiddenSize, 0);

        for (std::size_t t = 0; t < input.size(); ++t)
            hiddenStates.emplace_back(Tanh(
                m_inputToHidden * input[t] + m_hiddenToHidden * hiddenStates[t] + m_hiddenBias
            ));

        return hiddenStates;
    }

    math_vector<double> m_inputToHidden;
    matrix<double> m_hiddenToHidden;
    math_vector<double> m_hiddenBias,
                        m_hiddenToOutput;
    double m_outputBias;
    double m_learningRate;
    std::size_t m_hiddenSize;
};

#endif //RNNTRAFFICFORECASTING_RECURRENT_NEURAL_NETWORK_H