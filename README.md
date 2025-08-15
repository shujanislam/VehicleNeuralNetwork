# VehicleNeuralNetwork

A tiny C++ neural network that classifies sentences as **Car** or **Bike**.  
Uses **two hidden neurons per category** and simple weight updates to learn from example sentences. Perfect for learning the basics of neural networks without diving deep into AI frameworks.



## Features

- Simple feedforward neural network in C++
- Trains word weights from example sentences
- Two hidden neurons per category (Car and Bike)
- Predicts category based on sentence input

## Usage

1. Clone the repository
```bash 
git clone https://github.com/shujanislam/VehicleNeuralNetwork.git
```
2. Navigate to the folder and compile:
```bash 
g++ main.cpp -o VehicleNN
```
3. Run the executable file:
```bash 
./VehicleNN
```
4.  Enter a sentence, and the program will predict if it's a **Car** or **Bike**.
    

## Example

`Enter your input: I love riding my mountainbike
Prediction: Bike` 

`Enter your input: My machine has a sunroof
Prediction: Car`


## Learning Goals

-   Understand basic neural network concepts
    
-   Learn how word weights can be used for simple NLP classification
    
-   Practice implementing feedforward networks with minimal code
