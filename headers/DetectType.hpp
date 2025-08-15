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
extern float hidden1_car, hidden2_car;
extern float hidden1_bike, hidden2_bike;

// Activation function
inline float relu(float x) {
    return max(0.0f, x);
}

// Detect functions
inline float DetectCar(string input) {
    stringstream ss(input);
    string word;
    hidden1_car = hidden2_car = 0.0;

    while (ss >> word) {
        hidden1_car += car_weight[word];
        hidden2_car += car_weight[word] * 0.5;
    }

    hidden1_car = relu(hidden1_car);
    hidden2_car = relu(hidden2_car);

    return hidden1_car * 1.0 + hidden2_car * 0.8;
}

inline float DetectBike(string input) {
    stringstream ss(input);
    string word;
    hidden1_bike = hidden2_bike = 0.0;

    while (ss >> word) {
        hidden1_bike += bike_weight[word];
        hidden2_bike += bike_weight[word] * 0.5;
    }

    hidden1_bike = relu(hidden1_bike);
    hidden2_bike = relu(hidden2_bike);

    return hidden1_bike * 1.0 + hidden2_bike * 0.8;
}

#endif // DETECTTYPE_HPP
