#ifndef TRAINMODEL_HPP
#define TRAINMODEL_HPP

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include "DetectType.hpp"
using namespace std;

extern map<string, float> car_weight;
extern map<string, float> bike_weight;

// Forward declarations for Detect functions
float DetectCar(string input);
float DetectBike(string input);

// TrainModel function
inline void TrainModel() {
    vector<pair<string, string>> train_data = {
        {"I washed my car yesterday", "car"},
        {"My car has a sunroof", "car"},
        {"The engine of my car is loud", "car"},
        {"My four_wheeler broke down", "car"},
        {"I bought new tires for my car", "car"},
        {"My vehicle is parked outside", "car"},
        {"Cars are fun to drive", "car"},
        {"My car consumes a lot of fuel", "car"},
        {"I took my car for a long drive", "car"},
        {"The gears of my car are smooth", "car"},
        {"I love riding my mountainbike", "bike"},
        {"My pedals are broken on my bike", "bike"},
        {"I bought a new helmet for biking", "bike"},
        {"Bikes are easy to park", "bike"},
        {"My bike tires need air", "bike"},
        {"Superbikes are very fast", "bike"},
        {"I rode my bike uphill today", "bike"},
        {"I cleaned my bike yesterday", "bike"},
        {"Mountain bikes are good for trails", "bike"},
        {"Pedal harder to move faster", "bike"},
        {"The sedan is parked outside", "car"},
        {"I love my SUV for long trips", "car"},
        {"Hatchbacks are easy to drive", "car"},
        {"Trucks carry heavy loads", "car"},
        {"My scooter is very convenient", "bike"},
        {"I fixed the chain on my bike", "bike"},
        {"The sports car has new tires", "car"},
        {"I went off-road with my mountainbike", "bike"},
        {"I drove my truck to the city", "car"},
        {"The electric bike is very quiet", "bike"},
        {"My hybrid car saves fuel", "car"},
        {"I enjoyed a ride on my superbike", "bike"},
        {"I checked the brakes on my car", "car"},
        {"The bike needs oil for smooth riding", "bike"}
    };

    vector<string> car_words = {
        "car", "cars", "motor", "motor_vehicle", "four_wheeler", "vehicle",
        "gear", "gears", "tires", "fuel", "engine", "sunroof", "sedan",
        "suv", "hatchback", "truck", "hybrid", "sports", "brakes"
    };

    vector<string> bike_words = {
        "bike", "bikes", "superbike", "mountainbike", "pedals", "pedal",
        "gear", "gears", "tires", "fuel", "helmet", "scooter", "chain",
        "off-road", "electric", "ride", "brakes", "oil"
    };

    float learning_rate = 0.1;

    // Initialize all words in weight maps
    for (auto &sample : train_data) {
        stringstream ss(sample.first);
        string word;
        while (ss >> word) {
            if (car_weight.find(word) == car_weight.end()) car_weight[word] = 0.0;
            if (bike_weight.find(word) == bike_weight.end()) bike_weight[word] = 0.0;
        }
    }

    // Train car weights
    for (auto &sample : train_data) {
        stringstream ss(sample.first);
        string word;
        float car_score = DetectCar(sample.first);
        float target_car = (sample.second.find("car") != string::npos) ? 1.0 : 0.0;
        float error_car = target_car - car_score;

        while (ss >> word) {
            for (auto &w : car_words) {
                if (w == word) {
                    car_weight[word] += learning_rate * error_car;
                }
            }
        }
    }

    // Train bike weights
    for (auto &sample : train_data) {
        stringstream ss(sample.first);
        string word;
        float bike_score = DetectBike(sample.first);
        float target_bike = (sample.second.find("bike") != string::npos) ? 1.0 : 0.0;
        float error_bike = target_bike - bike_score;

        while (ss >> word) {
            for (auto &w : bike_words) {
                if (w == word) {
                    bike_weight[w] += learning_rate * error_bike;
                }
            }
        }
    }
}

#endif // TRAINMODEL_HPP
