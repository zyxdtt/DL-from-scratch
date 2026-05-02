#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>

using namespace std;
using namespace Eigen;

// Type aliases for brevity
using mat = MatrixXd;
using vec = VectorXd;
using rvec = RowVectorXd;
using Point = vector<double>;

// Abstract base class for all neural network layers
class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass: compute output from input
    virtual mat forward(const mat& input) = 0;

    // Backward pass: compute gradient w.r.t. input, and accumulate parameter gradients
    virtual mat backward(const mat& grad_output) = 0;

    // Update learnable parameters using accumulated gradients
    virtual void update_params(double learning_rate) {}

    // Return the name of this layer (for debugging)
    virtual string name() const = 0;
};

// ReLU activation layer
class ReLU : public Layer {
private:
    mat input_cache;  // Cached input for backward pass

public:
    mat forward(const mat& input) override {
        input_cache = input;
        return input.array().max(0.0).matrix();
    }

    mat backward(const mat& grad_output) override {
        // Gradient flows through only where input > 0
        mat mask = (input_cache.array() > 0.0).cast<double>();
        return grad_output.array() * mask.array();
    }

    string name() const override { return "ReLU"; }
};

// Fully connected (linear) layer
class Linear : public Layer {
private:
    mat weights;          // Weight matrix [in_dim, out_dim]
    rvec bias;            // Bias row vector [1, out_dim]
    mat input_cache;      // Cached input for backward pass
    mat grad_weights;     // Accumulated weight gradients
    rvec grad_bias;       // Accumulated bias gradients
    mat m_weights, v_weights;
    rvec m_bias, v_bias;
    double beta1, beta2, eps;
    int t;

public:
    Linear(int in_dim, int out_dim) :t(0), beta1(0.9), beta2(0.999), eps(1e-8) {
        // He initialization for ReLU
        weights = mat::Random(in_dim, out_dim) * sqrt(2.0 / in_dim);
        bias = rvec::Zero(out_dim);
        grad_weights = mat::Zero(in_dim, out_dim);
        grad_bias = rvec::Zero(out_dim);
        m_weights = mat::Zero(in_dim, out_dim);
        v_weights = mat::Zero(in_dim, out_dim);
        m_bias = rvec::Zero(out_dim);
        v_bias = rvec::Zero(out_dim);
    }

    mat forward(const mat& input) override {
        input_cache = input;
        mat output = input * weights;
        output.rowwise() += bias;  // Broadcast bias to each row
        return output;
    }

    mat backward(const mat& grad_output) override {
        // Accumulate gradients
        grad_weights += input_cache.transpose() * grad_output;
        grad_bias += grad_output.colwise().sum();

        // Return gradient w.r.t. input for previous layer
        return grad_output * weights.transpose();
    }

    void update_params(double learning_rate = 0.01) override {
        double max_norm = 1.0;
        if (grad_weights.norm() > max_norm)
            grad_weights *= max_norm / grad_weights.norm();
        if (grad_bias.norm() > max_norm)
            grad_bias *= max_norm / grad_bias.norm();
        t++;
        for (int i = 0; i < weights.rows(); i++) {
            for (int j = 0; j < weights.cols(); j++) {
                double g = grad_weights(i, j);
                m_weights(i, j) = beta1 * m_weights(i, j) + (1 - beta1) * g;
                v_weights(i, j) = beta2 * v_weights(i, j) + (1 - beta2) * g * g;
                double m_hat = m_weights(i, j) / (1 - pow(beta1, t));
                double v_hat = v_weights(i, j) / (1 - pow(beta2, t));
                weights(i, j) -= learning_rate * m_hat / (sqrt(v_hat) + eps);
            }
        }
        for (int j = 0; j < bias.cols(); j++) {
            double g = grad_bias(j);
            m_bias(j) = beta1 * m_bias(j) + (1 - beta1) * g;
            v_bias(j) = beta2 * v_bias(j) + (1 - beta2) * g * g;
            double m_hat = m_bias(j) / (1 - pow(beta1, t));
            double v_hat = v_bias(j) / (1 - pow(beta2, t));
            bias(j) -= learning_rate * m_hat / (sqrt(v_hat) + eps);
        }

        // Zero gradients after update
        grad_weights.setZero();
        grad_bias.setZero();
    }

    string name() const override { return "Linear"; }
};

// Softmax activation layer (output probabilities)
class softmax : public Layer {
public:
    mat forward(const mat& input) override {
        // Numerically stable softmax: subtract max before exp
        mat shifted = input.colwise() - input.rowwise().maxCoeff();
        mat exp_vals = shifted.array().exp();
        mat probabilities = exp_vals.array().colwise() / exp_vals.rowwise().sum().array();
        return probabilities;
    }

    mat backward(const mat& grad_output) override {
        // When combined with cross-entropy loss, gradient passes through directly
        return grad_output;
    }

    string name() const override { return "softmax"; }
};

// Feedforward Neural Network class
class FNN {
private:
    vector<Layer*> layers;      // Stack of layers
    int number_of_train;        // Number of training samples
    int number_of_dimension;    // Input feature dimension
    int number_of_class;        // Number of classes (for classification)
    bool is_fitted;             // Whether the model has been trained

    // Single training step for regression (MSE loss)
    double one_step_train_regression(const mat& X_batch,
        const mat& y_batch,
        double learning_rate) {
        mat pred = forward(X_batch);
        mat diff = pred - y_batch;
        mat grad_loss = diff / X_batch.rows();
        backward(grad_loss);
        update_params(learning_rate);
        return diff.array().square().sum();
    }

    // Single training step for classification (Cross-Entropy loss)
    double one_step_train_classify(const mat& X_batch,
        const mat& y_batch,
        double learning_rate) {
        mat probabilities = forward(X_batch);
        double eps = 1e-12;
        double loss = -(y_batch.array() * (probabilities.array() + eps).log()).sum() / X_batch.rows();
        mat grad_loss = (probabilities - y_batch) / X_batch.rows();
        backward(grad_loss);
        update_params(learning_rate);
        return loss;
    }

    // Forward pass through all layers
    mat forward(const mat& inputs) {
        auto input = inputs;
        for (auto layer : layers) {
            input = layer->forward(input);
        }
        return input;
    }

    // Backward pass through all layers
    void backward(const mat& grad_loss) {
        mat grad = grad_loss;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers[i]->backward(grad);
        }
    }

    // Update parameters of all layers
    void update_params(double learning_rate = 0.2) {
        for (auto layer : layers) {
            layer->update_params(learning_rate);
        }
    }

    // Forward pass excluding the final softmax layer (returns logits)
    mat forward_softmax(const mat& inputs) {
        auto input = inputs;
        for (int i = 0; i < layers.size() - 1; i++) {  // Skip softmax layer
            input = layers[i]->forward(input);
        }
        return input;
    }

public:
    ~FNN() {
        for (auto layer : layers) delete layer;
    }

    FNN() : is_fitted(false) {}

    // Add a layer to the network
    void add(Layer* layer) {
        layers.push_back(layer);
    }

    // Train the network for regression tasks
    void fit_regression(const vector<Point>& X_train,
        const vector<double>& y_train,
        int epochs = 50,
        double learning_rate = 0.2,
        int batch_size = 32,
        bool verbose = true) {
        is_fitted = true;
        number_of_train = X_train.size();
        number_of_dimension = X_train[0].size();

        // Build training data matrix [X | y]
        mat train_data(number_of_train, number_of_dimension + 1);
        for (int i = 0; i < number_of_train; i++) {
            train_data.row(i) = rvec::Map(X_train[i].data(), number_of_dimension + 1);
            train_data(i, number_of_dimension) = y_train[i];
        }

        random_device rd;
        auto seed = rd();
        mt19937 gen(seed);
        vector<int> batch_seq(number_of_train);
        iota(batch_seq.begin(), batch_seq.end(), 0);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            shuffle(batch_seq.begin(), batch_seq.end(), gen);
            double epoch_loss = 0.0;

            for (int i = 0; i < number_of_train; i += batch_size) {
                int end = min(i + batch_size, number_of_train);
                int number_of_batch = end - i;

                mat batch_data(number_of_batch, number_of_dimension + 1);
                for (int j = 0; j < number_of_batch; j++) {
                    batch_data.row(j) = train_data.row(batch_seq[i + j]);
                }

                auto X_batch = batch_data.leftCols(number_of_dimension);
                auto y_batch = batch_data.col(number_of_dimension);
                epoch_loss += one_step_train_regression(X_batch, y_batch, learning_rate);
            }

            if (verbose) {
                cout << "Epoch " << epoch << ", Loss: " << epoch_loss / number_of_train << '\n';
            }
        }
    }

    // Predict continuous values for regression
    vector<double> predict_regression(const vector<Point>& X_test) {
        vector<double> y_test;
        if (!is_fitted) return y_test;

        int number_of_test = X_test.size();
        y_test.resize(number_of_test);

        mat X(number_of_test, number_of_dimension);
        for (int i = 0; i < number_of_test; i++) {
            X.row(i) = rvec::Map(X_test[i].data(), number_of_dimension);
        }

        mat pred = forward(X);
        for (int i = 0; i < number_of_test; i++) {
            y_test[i] = pred(i, 0);
        }
        return y_test;
    }

    // Train the network for classification tasks
    void fit_classify(const vector<Point>& X_train,
        const vector<int>& y_train,
        int number_of_class = -1,
        int epochs = 50,
        double learning_rate = 0.01,
        int batch_size = 32,
        bool verbose = true) {
        is_fitted = true;
        number_of_train = X_train.size();
        number_of_dimension = X_train[0].size();

        // Infer number of classes if not provided
        if (number_of_class == -1) {
            unordered_set<int> cnt;
            for (int label : y_train) cnt.insert(label);
            number_of_class = cnt.size();
        }
        this->number_of_class = number_of_class;

        // Convert integer labels to one-hot encoding
        mat y_onehot = mat::Zero(number_of_train, number_of_class);
        for (int i = 0; i < number_of_train; i++) {
            if (y_train[i] >= 0 && y_train[i] < number_of_class) {
                y_onehot(i, y_train[i]) = 1.0;
            }
        }

        // Build training data matrix [X | one-hot y]
        mat train_data(number_of_train, number_of_dimension + number_of_class);
        for (int i = 0; i < number_of_train; i++) {
            train_data.row(i).head(number_of_dimension) = rvec::Map(X_train[i].data(), number_of_dimension);
            train_data.row(i).tail(number_of_class) = y_onehot.row(i);
        }

        random_device rd;
        mt19937 gen(rd());
        vector<int> batch_seq(number_of_train);
        iota(batch_seq.begin(), batch_seq.end(), 0);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            shuffle(batch_seq.begin(), batch_seq.end(), gen);
            double epoch_loss = 0.0;

            for (int i = 0; i < number_of_train; i += batch_size) {
                int end = min(i + batch_size, number_of_train);
                int number_of_batch = end - i;

                mat batch_data(number_of_batch, number_of_dimension + number_of_class);
                for (int j = 0; j < number_of_batch; j++) {
                    batch_data.row(j) = train_data.row(batch_seq[i + j]);
                }

                mat X_batch = batch_data.leftCols(number_of_dimension);
                mat y_batch = batch_data.rightCols(number_of_class);
                epoch_loss += one_step_train_classify(X_batch, y_batch, learning_rate);
            }

            if (verbose) {
                cout << "Epoch " << epoch << ", Loss: " << epoch_loss / number_of_train << '\n';
            }
        }
    }

    // Predict class labels for classification
    vector<int> predict_classify(const vector<Point>& X_test) {
        vector<int> y_test;
        if (!is_fitted) return y_test;

        int number_of_test = X_test.size();
        y_test.resize(number_of_test);

        mat X(number_of_test, number_of_dimension);
        for (int i = 0; i < number_of_test; i++) {
            X.row(i) = rvec::Map(X_test[i].data(), number_of_dimension);
        }

        mat logits = forward_softmax(X);
        for (int i = 0; i < number_of_test; i++) {
            logits.row(i).maxCoeff(&y_test[i]);
        }
        return y_test;
    }

    // Predict class probabilities for classification
    vector<Point> predict_proba(const vector<Point>& X_test) {
        vector<Point> y_test;
        if (!is_fitted) return y_test;

        int number_of_test = X_test.size();
        y_test.resize(number_of_test);

        mat X(number_of_test, number_of_dimension);
        for (int i = 0; i < number_of_test; i++) {
            X.row(i) = rvec::Map(X_test[i].data(), number_of_dimension);
        }

        mat probabilities = forward(X);
        for (int i = 0; i < number_of_test; i++) {
            Point prob_row(number_of_class);
            for (int j = 0; j < number_of_class; j++) {
                prob_row[j] = probabilities(i, j);
            }
            y_test[i] = std::move(prob_row);
        }
        return y_test;
    }

    // Print the network architecture
    void print_structure() const {
        cout << "Network Structure:\n";
        for (size_t i = 0; i < layers.size(); ++i) {
            std::cout << "  " << i << ": " << layers[i]->name() << "\n";
        }
    }

    // Return the number of layers
    size_t size() const {
        return layers.size();
    }
};