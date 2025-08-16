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
inline float DetectVehicle(string input, map<string, float>& weights, 
                           float& h1, float& h2, float& h3, float& h4, float& h5) {
    transform(input.begin(), input.end(), input.begin(), ::tolower);
    stringstream ss(input);
    string word;
    h1 = h2 = h3 = h4 = h5 = 0.0f;
    bool negation = false;

    while (ss >> word) {
        if (word == "not" || word == "never" || word == "don't" || word == "no" || word == "not a") {
            negation = true;
            continue;
        }
       
        if (word == "a" && negation) {
            string next;
            if (ss >> next) word += " " + next; // join for lookup
        }
        
        float w = weights[word];
        if (negation) {
            w = -w;
            negation = false;
        }

        h1 += w;
        h2 += w * 0.6f;
        h3 += w * 0.4f;
        h4 += w * 0.3f;
        h5 += w * 0.2f;
    }

    h1 = relu(h1); h2 = relu(h2); h3 = relu(h3); h4 = relu(h4); h5 = relu(h5);

    float output = h1 * 1.0f + h2 * 0.8f + h3 * 0.5f + h4 * 0.3f + h5 * 0.2f;
    return sigmoid(output);
}

inline float DetectCar(string input) {
    return DetectVehicle(input, car_weight, hidden1_car, hidden2_car, hidden3_car, hidden4_car, hidden5_car);
}

inline float DetectBike(string input) {
    return DetectVehicle(input, bike_weight, hidden1_bike, hidden2_bike, hidden3_bike, hidden4_bike, hidden5_bike);
}

#endif
