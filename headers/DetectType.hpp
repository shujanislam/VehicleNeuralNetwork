#ifndef DETECTTYPE_HPP
#define DETECTTYPE_HPP

#include <iostream>
#include <sstream>
#include <map>
#include <algorithm>
#include <cmath>
using namespace std;

extern map<string, float> car_weight;
extern map<string, float> bike_weight;

// Hidden layer neurons (extern so they can be shared)
extern float hidden1_car, hidden2_car, hidden3_car, hidden4_car, hidden5_car;
extern float hidden1_bike, hidden2_bike, hidden3_bike, hidden4_bike, hidden5_bike;

// Activation functions
inline float relu(float x) {
    return max(0.0f, x);
}

inline float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// Detect functions
inline float DetectCar(string input) {
    stringstream ss(input);
    string word;
    hidden1_car = hidden2_car = hidden3_car = hidden4_car = hidden5_car = 0.0;

    while (ss >> word) {
        hidden1_car += car_weight[word];
        hidden2_car += car_weight[word] * 0.6;
        hidden3_car += car_weight[word] * 0.4;
        hidden4_car += car_weight[word] * 0.3;
        hidden5_car += car_weight[word] * 0.2;
    }

    // Hidden activations (ReLU or Sigmoid – you can switch)
    hidden1_car = relu(hidden1_car);
    hidden2_car = relu(hidden2_car);
    hidden3_car = relu(hidden3_car);
    hidden4_car = relu(hidden4_car);
    hidden5_car = relu(hidden5_car);

    // Final output layer with sigmoid
    float output = hidden1_car * 1.0 + hidden2_car * 0.8 + hidden3_car * 0.5 + hidden4_car * 0.3 + hidden5_car * 0.2;
    return sigmoid(output);
}

inline float DetectBike(string input) {
    stringstream ss(input);
    string word;
    hidden1_bike = hidden2_bike = hidden3_bike = hidden4_bike = hidden5_bike = 0.0;

    while (ss >> word) {
        hidden1_bike += bike_weight[word];
        hidden2_bike += bike_weight[word] * 0.6;
        hidden3_bike += bike_weight[word] * 0.4;
        hidden4_bike += bike_weight[word] * 0.3;
        hidden5_bike += bike_weight[word] * 0.2;
    }

    hidden1_bike = relu(hidden1_bike);
    hidden2_bike = relu(hidden2_bike);
    hidden3_bike = relu(hidden3_bike);
    hidden4_bike = relu(hidden4_bike);
    hidden5_bike = relu(hidden5_bike);

    float output = hidden1_bike * 1.0 + hidden2_bike * 0.8 + hidden3_bike * 0.5 + hidden4_bike * 0.3 + hidden5_bike * 0.2;
    return sigmoid(output);
}

#endif
