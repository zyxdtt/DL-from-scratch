// test_CNN.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "CNN.hpp"

using namespace std;
using namespace Eigen;

pair<tensor4d, vector<int>> load_mnist(const string& img_file, const string& lbl_file, int N, int rows = 28, int cols = 28) {
    ifstream img_fs(img_file), lbl_fs(lbl_file);
    if (!img_fs.is_open() || !lbl_fs.is_open()) throw runtime_error("Cannot open data files");
    tensor4d data(N, tensor3d(1, mat(rows, cols)));
    vector<int> labels(N);
    for (int i = 0; i < N; ++i) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                img_fs >> data[i][0](r, c);
        lbl_fs >> labels[i];
    }
    return { data, labels };
}

tensor4d one_hot(const vector<int>& labels, int C) {
    tensor4d oh(labels.size(), tensor3d(C, mat(1, 1)));
    for (size_t i = 0; i < labels.size(); ++i)
        for (int c = 0; c < C; ++c)
            oh[i][c](0, 0) = (labels[i] == c);
    return oh;
}

tensor4d softmax(const tensor4d& logits) {
    tensor4d probs = logits;
    for (auto& sample : probs) {
        double max_val = -1e9, sum = 0;
        for (auto& cls : sample) max_val = max(max_val, cls(0, 0));
        for (auto& cls : sample) {
            double e = exp(cls(0, 0) - max_val);
            cls(0, 0) = e;
            sum += e;
        }
        for (auto& cls : sample) cls(0, 0) /= sum;
    }
    return probs;
}

double cross_entropy(const tensor4d& probs, const tensor4d& targets) {
    double loss = 0;
    int B = probs.size(), C = probs[0].size();
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < C; ++j)
            loss += targets[i][j](0, 0) * log(probs[i][j](0, 0) + 1e-8);
    return -loss / B;
}

tensor4d softmax_cross_entropy_grad(const tensor4d& logits, const tensor4d& targets) {
    tensor4d grad = softmax(logits);
    for (size_t i = 0; i < grad.size(); ++i)
        for (size_t j = 0; j < grad[i].size(); ++j)
            grad[i][j](0, 0) -= targets[i][j](0, 0);
    return grad;
}

double accuracy(const tensor4d& probs, const vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        int pred = 0;
        double best = probs[i][0](0, 0);
        for (size_t j = 1; j < probs[i].size(); ++j)
            if (probs[i][j](0, 0) > best) best = probs[i][j](0, 0), pred = j;
        if (pred == labels[i]) ++correct;
    }
    return double(correct) / probs.size();
}

int main() {
    const int BATCH = 32, EPOCHS = 5, CLASSES = 10;
    const double LR = 0.001;

    cout << "Loading MNIST..." << endl;
    auto [X_train, y_train] = load_mnist("X_train.txt", "y_train.txt", 60000);
    auto [X_test, y_test] = load_mnist("X_test.txt", "y_test.txt", 10000);
    cout << "Train: " << X_train.size() << ", Test: " << X_test.size() << endl;

    vector<unique_ptr<Layer>> layers;
    layers.emplace_back(new Conv(3, 1, 1, 16, 28, 28, 1));
    layers.emplace_back(new ReLu);
    layers.emplace_back(new MaxPooling(2, 2, 0, 28, 28));
    layers.emplace_back(new Conv(3, 1, 1, 32, 14, 14, 16));
    layers.emplace_back(new ReLu);
    layers.emplace_back(new MaxPooling(2, 2, 0, 14, 14));
    layers.emplace_back(new Flatten);
    layers.emplace_back(new Linear(1568, 128));
    layers.emplace_back(new ReLu);
    layers.emplace_back(new Linear(128, CLASSES));

    cout << "Start training..." << endl;
    int num_batches = X_train.size() / BATCH;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start = chrono::high_resolution_clock::now();
        double total_loss = 0.0;

        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * BATCH;
            tensor4d batch_input, batch_target;
            for (int i = 0; i < BATCH; ++i) {
                batch_input.push_back(X_train[start_idx + i]);
                batch_target.push_back(one_hot({ y_train[start_idx + i] }, CLASSES)[0]);
            }

            tensor4d x = batch_input;
            for (auto& layer : layers)
                x = layer->forward(x);

            double loss = cross_entropy(softmax(x), batch_target);
            total_loss += loss;

            tensor4d grad = softmax_cross_entropy_grad(x, batch_target);
            for (int i = layers.size() - 1; i >= 0; --i)
                grad = layers[i]->backward(grad);

            for (auto& layer : layers)
                layer->update_params(LR);
        }

        auto end = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(end - start).count();

        tensor4d test_input(X_test.begin(), X_test.begin() + 1000);
        tensor4d tx = test_input;
        for (auto& layer : layers) tx = layer->forward(tx);
        double acc = accuracy(softmax(tx), vector<int>(y_test.begin(), y_test.begin() + 1000));

        cout << "Epoch " << epoch + 1 << "/" << EPOCHS
            << " | loss: " << total_loss / num_batches
            << " | test acc: " << acc * 100 << "%"
            << " | time: " << elapsed << "s" << endl;
    }

    tensor4d tx = X_test;
    for (auto& layer : layers) tx = layer->forward(tx);
    double final_acc = accuracy(softmax(tx), y_test);
    cout << "\nFinal test accuracy: " << final_acc * 100 << "%" << endl;

    return 0;
}