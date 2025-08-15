#ifndef DETECTTYPE_HPP
#define DETECTTYPE_HPP

#include <iostream>
#include <sstream>
#include <map>
#include <algorithm>
using namespace std;

extern map<string, float> car_weight;
extern map<string, float> bike_weight;

// Hidden layer neurons (extern so they can be shared)
extern float hidden1_car, hidden2_car, hidden3_car;
extern float hidden1_bike, hidden2_bike, hidden3_bike;

// Activation function
inline float relu(float x) {
    return max(0.0f, x);
}

// Derivative of ReLU for backprop
inline float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// Detect functions
inline float DetectCar(string input) {
    stringstream ss(input);
    string word;
    hidden1_car = hidden2_car = hidden3_car = 0.0;

    while (ss >> word) {
        hidden1_car += car_weight[word];
        hidden2_car += car_weight[word] * 0.5;
        hidden3_car += car_weight[word] * 0.3;
    }

    hidden1_car = relu(hidden1_car);
    hidden2_car = relu(hidden2_car);
    hidden3_car = relu(hidden3_car);

    // Return final output
    return hidden1_car * 1.0 + hidden2_car * 0.8 + hidden3_car * 0.5;
}

inline float DetectBike(string input) {
    stringstream ss(input);
    string word;
    hidden1_bike = hidden2_bike = hidden3_bike = 0.0;

    while (ss >> word) {
        hidden1_bike += bike_weight[word];
        hidden2_bike += bike_weight[word] * 0.5;
        hidden3_bike += bike_weight[word] * 0.3;
    }

    hidden1_bike = relu(hidden1_bike);
    hidden2_bike = relu(hidden2_bike);
    hidden3_bike = relu(hidden3_bike);

    // Return final output
    return hidden1_bike * 1.0 + hidden2_bike * 0.8 + hidden3_bike * 0.5;
}

// Function to get gradients for backprop
inline void ComputeCarGradients(string input, float output_error, map<string, float>& gradients) {
    stringstream ss(input);
    string word;

    // Weighted contribution of each hidden neuron
    float grad1 = output_error * 1.0f * relu_derivative(hidden1_car);
    float grad2 = output_error * 0.8f * relu_derivative(hidden2_car);
    float grad3 = output_error * 0.5f * relu_derivative(hidden3_car);

    while (ss >> word) {
        gradients[word] += grad1 + grad2 * 0.5f + grad3 * 0.3f;
    }
}

inline void ComputeBikeGradients(string input, float output_error, map<string, float>& gradients) {
    stringstream ss(input);
    string word;

    float grad1 = output_error * 1.0f * relu_derivative(hidden1_bike);
    float grad2 = output_error * 0.8f * relu_derivative(hidden2_bike);
    float grad3 = output_error * 0.5f * relu_derivative(hidden3_bike);

    while (ss >> word) {
        gradients[word] += grad1 + grad2 * 0.5f + grad3 * 0.3f;
    }
}

#endif // DETECTTYPE_HPP
