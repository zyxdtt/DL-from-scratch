#include "FNN.hpp"
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>

vector<Point> read_csv_features(const string& filename) {
    vector<Point> data;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        Point row;
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        data.push_back(row);
    }
    return data;
}

vector<double> read_csv_target(const string& filename) {
    vector<double> data;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        data.push_back(stod(line));
    }
    return data;
}

double mse(const vector<double>& y_true, const vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        sum += pow(y_true[i] - y_pred[i], 2);
    }
    return sum / y_true.size();
}

int main() {
    cout << "=== California Housing Price Prediction (C++ MLP) ===" << endl;

    // 1. Load data
    cout << "Loading data..." << endl;
    vector<Point> X_train = read_csv_features("X_train.csv");
    vector<double> y_train = read_csv_target("y_train.csv");
    vector<Point> X_test = read_csv_features("X_test.csv");
    vector<double> y_test = read_csv_target("y_test.csv");

    cout << "Train set: " << X_train.size() << " x " << X_train[0].size() << endl;
    cout << "Test set: " << X_test.size() << " x " << X_test[0].size() << endl;

    // 2. Build network
    FNN net;
    net.add(new Linear(8, 64));
    net.add(new ReLU());
    net.add(new Linear(64, 32));
    net.add(new ReLU());
    net.add(new Linear(32, 1));

    net.print_structure();

    // 3. Train
    cout << "\nTraining started..." << endl;
    auto start = chrono::high_resolution_clock::now();

    net.fit_regression(X_train, y_train, 300, 0.01, 64, true);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Training time: " << duration.count() << " seconds" << endl;

    // 4. Evaluate
    vector<double> pred_train = net.predict_regression(X_train);
    vector<double> pred_test = net.predict_regression(X_test);

    cout << "\n=== Evaluation Results ===" << endl;
    cout << "Train MSE: " << mse(y_train, pred_train) << endl;
    cout << "Test MSE: " << mse(y_test, pred_test) << endl;

    // 5. Sample predictions
    cout << "\n=== Sample Predictions (first 5 test samples) ===" << endl;
    for (int i = 0; i < 5 && i < (int)y_test.size(); i++) {
        cout << "True: " << y_test[i] << ", Pred: " << pred_test[i] << endl;
    }

    return 0;
}