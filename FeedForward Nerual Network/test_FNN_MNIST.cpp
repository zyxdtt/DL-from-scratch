#include "FNN.hpp"
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

// Read CSV features
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

// Read CSV labels (integers)
vector<int> read_csv_labels(const string& filename) {
    vector<int> data;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        data.push_back(stoi(line));
    }
    return data;
}

// Calculate accuracy
double accuracy(const vector<int>& y_true, const vector<int>& y_pred) {
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (y_true[i] == y_pred[i]) correct++;
    }
    return 100.0 * correct / y_true.size();
}

int main() {
    cout << "=== MNIST Classification (C++ MLP) ===" << endl;
    cout << fixed << setprecision(4);

    // 1. Load data
    cout << "\nLoading data..." << endl;
    vector<Point> X_train = read_csv_features("mnist_X_train.csv");
    vector<int> y_train = read_csv_labels("mnist_y_train.csv");
    vector<Point> X_test = read_csv_features("mnist_X_test.csv");
    vector<int> y_test = read_csv_labels("mnist_y_test.csv");

    cout << "Train set: " << X_train.size() << " x " << X_train[0].size() << endl;
    cout << "Test set: " << X_test.size() << " x " << X_test[0].size() << endl;
    cout << "Classes: 10 (digits 0-9)" << endl;

    // 2. Build network (784 -> 256 -> 128 -> 10)
    FNN net;
    net.add(new Linear(784, 256));
    net.add(new ReLU());
    net.add(new Linear(256, 128));
    net.add(new ReLU());
    net.add(new Linear(128, 10));
    net.add(new softmax());

    net.print_structure();

    // 3. Train
    cout << "\nTraining started..." << endl;
    auto start = chrono::high_resolution_clock::now();

    // Use smaller epoch count for initial testing
    net.fit_classify(X_train, y_train, 10, 50, 0.01, 64, true);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Training time: " << duration.count() << " seconds" << endl;

    // 4. Evaluate
    cout << "\nEvaluating on training set..." << endl;
    vector<int> pred_train = net.predict_classify(X_train);
    double train_acc = accuracy(y_train, pred_train);
    cout << "Train Accuracy: " << train_acc << "%" << endl;

    cout << "\nEvaluating on test set..." << endl;
    vector<int> pred_test = net.predict_classify(X_test);
    double test_acc = accuracy(y_test, pred_test);
    cout << "Test Accuracy: " << test_acc << "%" << endl;

    // 5. Sample predictions with probabilities
    cout << "\n=== Sample Predictions (first 10 test samples) ===" << endl;
    cout << "True\tPred\tProbabilities (top 3)" << endl;
    cout << "----------------------------------------" << endl;

    vector<Point> proba_test = net.predict_proba(X_test);
    for (int i = 0; i < 10 && i < (int)y_test.size(); i++) {
        cout << y_test[i] << "\t" << pred_test[i] << "\t[";

        // Get top 3 probabilities
        vector<pair<double, int>> probs;
        for (int j = 0; j < 10; j++) {
            probs.push_back({ proba_test[i][j], j });
        }
        sort(probs.begin(), probs.end(), greater<pair<double, int>>());

        for (int j = 0; j < 3; j++) {
            cout << probs[j].second << ":" << fixed << setprecision(3) << probs[j].first;
            if (j < 2) cout << ", ";
        }
        cout << "]" << endl;
    }

    // 6. Summary
    cout << "\n=== Summary ===" << endl;
    cout << "Network: 784 -> 256 -> 128 -> 10" << endl;
    cout << "Train Accuracy: " << train_acc << "%" << endl;
    cout << "Test Accuracy: " << test_acc << "%" << endl;

    return 0;
}