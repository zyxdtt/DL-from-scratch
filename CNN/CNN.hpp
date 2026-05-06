//2026 5 5
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <string>
#include <limits>

using namespace std;
using namespace Eigen;
using mat = MatrixXd;
using vec = VectorXd;
using rvec = RowVectorXd;
using Point = vector<double>;
using tensor3d = vector<mat>;
using tensor4d = vector<tensor3d>;

class Layer {
public:
    virtual ~Layer() = default;
    virtual tensor4d forward(const tensor4d& input) = 0;
    virtual tensor4d backward(const tensor4d& grad_output) = 0;
    virtual void update_params(double learning_rate) = 0;
    virtual string name() const = 0;
};

class Conv : public Layer {
private:
    tensor4d filters;      // [out_channels][in_channels][patch][patch]
    tensor4d input_cache;  // [batch][in_channels][H][W]
    tensor4d grad;         // gradient of filters, same shape as filters
    int stride;
    int padding;
    int in_channels;
    int in_rows, in_cols;
    int patch;
    int in_rows_pad, in_cols_pad;
    int out_rows, out_cols;

    mat apply_padding(const mat& img) const {
        if (padding == 0) return img;
        mat padded = mat::Zero(in_rows_pad, in_cols_pad);
        padded.block(padding, padding, in_rows, in_cols) = img;
        return padded;
    }

    // Convolution for a single filter on a single sample (multi-channel)
    mat conv_3d(const tensor3d& feature_map, const tensor3d& kernel) const {
        mat output = mat::Zero(out_rows, out_cols);
        for (int c = 0; c < in_channels; ++c) {
            mat fm_pad = apply_padding(feature_map[c]);
            const mat& k = kernel[c];   // patch x patch
            for (int i = 0; i < out_rows; ++i) {
                for (int j = 0; j < out_cols; ++j) {
                    int row = i * stride;
                    int col = j * stride;
                    output(i, j) += fm_pad.block(row, col, patch, patch).cwiseProduct(k).sum();
                }
            }
        }
        return output;
    }

public:
    Conv(int patch, int stride, int padding, int num_filters,
        int input_rows, int input_cols, int channels)
        : stride(stride), padding(padding), in_channels(channels),
        in_rows(input_rows), in_cols(input_cols), patch(patch) {
        in_rows_pad = in_rows + 2 * padding;
        in_cols_pad = in_cols + 2 * padding;
        out_rows = (in_rows_pad - patch) / stride + 1;
        out_cols = (in_cols_pad - patch) / stride + 1;
        // He uniform initialization
        double fan_in = channels * patch * patch;
        double scale = sqrt(6.0 / fan_in);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dist(-scale, scale);
        filters.resize(num_filters);
        for (int f = 0; f < num_filters; ++f) {
            filters[f].resize(channels);
            for (int c = 0; c < channels; ++c) {
                filters[f][c].resize(patch, patch);
                for (int i = 0; i < patch; ++i) {
                    for (int j = 0; j < patch; ++j) {
                        filters[f][c](i, j) = dist(gen);
                    }
                }
            }
        }
    }

    tensor4d forward(const tensor4d& input) override {
        input_cache = input;
        int batch_size = input.size();
        int num_filters = filters.size();
        tensor4d output(batch_size, tensor3d(num_filters, mat(out_rows, out_cols)));
        for (int b = 0; b < batch_size; ++b) {
            for (int f = 0; f < num_filters; ++f) {
                output[b][f] = conv_3d(input[b], filters[f]);
            }
        }
        return output;
    }

    tensor4d backward(const tensor4d& grad_output) override {
        int batch_size = grad_output.size();
        int num_filters = filters.size();
        // Initialize filter gradients
        grad.resize(num_filters);
        for (int f = 0; f < num_filters; ++f) {
            grad[f].resize(in_channels);
            for (int c = 0; c < in_channels; ++c) {
                grad[f][c] = mat::Zero(patch, patch);
            }
        }
        // Input gradient (same shape as input_cache)
        tensor4d grad_input(batch_size, tensor3d(in_channels, mat::Zero(in_rows, in_cols)));
        for (int b = 0; b < batch_size; ++b) {
            for (int f = 0; f < num_filters; ++f) {
                for (int c = 0; c < in_channels; ++c) {
                    // Padded input used for patch extraction
                    mat fm_pad = apply_padding(input_cache[b][c]);
                    for (int i = 0; i < out_rows; ++i) {
                        for (int j = 0; j < out_cols; ++j) {
                            int row = i * stride;
                            int col = j * stride;
                            mat patch_mat = fm_pad.block(row, col, patch, patch);
                            double grad_val = grad_output[b][f](i, j);
                            // Filter gradient
                            grad[f][c] += patch_mat * grad_val;
                            // Input gradient (propagate to original unpadded positions)
                            int orig_row_start = row - padding;
                            int orig_col_start = col - padding;
                            for (int kr = 0; kr < patch; ++kr) {
                                for (int kc = 0; kc < patch; ++kc) {
                                    int orig_r = orig_row_start + kr;
                                    int orig_c = orig_col_start + kc;
                                    if (orig_r >= 0 && orig_r < in_rows &&
                                        orig_c >= 0 && orig_c < in_cols) {
                                        grad_input[b][c](orig_r, orig_c) +=
                                            filters[f][c](kr, kc) * grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return grad_input;
    }

    void update_params(double learning_rate) override {
        int batch_size = input_cache.size();
        double factor = learning_rate / batch_size;
        for (int f = 0; f < (int)grad.size(); ++f) {
            for (int c = 0; c < (int)grad[f].size(); ++c) {
                filters[f][c] -= factor * grad[f][c];
            }
        }
    }

    string name() const override { return "Conv"; }
};

class ReLu : public Layer {
private:
    tensor4d input_cache;
public:
    tensor4d forward(const tensor4d& input) override {
        input_cache = input;
        tensor4d output = input;
        for (auto& batch : output) {
            for (auto& channel : batch) {
                channel = channel.cwiseMax(0.0);
            }
        }
        return output;
    }

    tensor4d backward(const tensor4d& grad_output) override {
        tensor4d grad_input = grad_output;
        for (size_t b = 0; b < grad_output.size(); ++b) {
            for (size_t c = 0; c < grad_output[0].size(); ++c) {
                mat mask = (input_cache[b][c].array() > 0.0).cast<double>();
                grad_input[b][c] = grad_input[b][c].array() * mask.array();
            }
        }
        return grad_input;
    }

    void update_params(double) override {}
    string name() const override { return "ReLu"; }
};

class MaxPooling : public Layer {
private:
    tensor4d input_cache;
    int patch;
    int stride;
    int padding;
    int in_rows, in_cols;
    int in_rows_pad, in_cols_pad;
    int out_rows, out_cols;

    mat apply_padding(const mat& img) const {
        if (padding == 0) return img;
        mat padded = mat::Constant(in_rows_pad, in_cols_pad, -1e9);
        padded.block(padding, padding, in_rows, in_cols) = img;
        return padded;
    }

public:
    MaxPooling(int patch, int stride, int padding, int in_rows, int in_cols)
        : patch(patch), stride(stride), padding(padding),
        in_rows(in_rows), in_cols(in_cols) {
        in_rows_pad = in_rows + 2 * padding;
        in_cols_pad = in_cols + 2 * padding;
        out_rows = (in_rows_pad - patch) / stride + 1;
        out_cols = (in_cols_pad - patch) / stride + 1;
    }

    tensor4d forward(const tensor4d& input) override {
        input_cache = input;
        tensor4d output(input.size(), tensor3d(input[0].size(), mat(out_rows, out_cols)));
        for (size_t b = 0; b < input.size(); ++b) {
            for (size_t c = 0; c < input[0].size(); ++c) {
                mat fm_pad = apply_padding(input[b][c]);
                for (int i = 0; i < out_rows; ++i) {
                    for (int j = 0; j < out_cols; ++j) {
                        int row = i * stride;
                        int col = j * stride;
                        mat block = fm_pad.block(row, col, patch, patch);
                        output[b][c](i, j) = block.maxCoeff();
                    }
                }
            }
        }
        return output;
    }

    tensor4d backward(const tensor4d& grad_output) override {
        tensor4d grad_input(input_cache.size(),
            tensor3d(input_cache[0].size(),
                mat::Zero(in_rows, in_cols)));
        for (size_t b = 0; b < grad_output.size(); ++b) {
            for (size_t c = 0; c < grad_output[0].size(); ++c) {
                mat fm_pad = apply_padding(input_cache[b][c]);
                for (int i = 0; i < out_rows; ++i) {
                    for (int j = 0; j < out_cols; ++j) {
                        int row = i * stride;
                        int col = j * stride;
                        mat block = fm_pad.block(row, col, patch, patch);
                        double max_val = block.maxCoeff();
                        // Find position of max (only first occurrence matters)
                        bool found = false;
                        int max_row = 0, max_col = 0;
                        for (int r = 0; r < patch && !found; ++r) {
                            for (int cr = 0; cr < patch; ++cr) {
                                if (block(r, cr) == max_val) {
                                    max_row = r; max_col = cr;
                                    found = true;
                                    break;
                                }
                            }
                        }
                        int orig_r = row + max_row - padding;
                        int orig_c = col + max_col - padding;
                        if (orig_r >= 0 && orig_r < in_rows &&
                            orig_c >= 0 && orig_c < in_cols) {
                            grad_input[b][c](orig_r, orig_c) += grad_output[b][c](i, j);
                        }
                    }
                }
            }
        }
        return grad_input;
    }

    void update_params(double) override {}
    string name() const override { return "MaxPooling"; }
};

class MeanPooling : public Layer {
private:
    tensor4d input_cache;
    int patch;
    int stride;
    int padding;
    int in_rows, in_cols;
    int in_rows_pad, in_cols_pad;
    int out_rows, out_cols;

    mat apply_padding(const mat& img) const {
        if (padding == 0) return img;
        mat padded = mat::Zero(in_rows_pad, in_cols_pad);
        padded.block(padding, padding, in_rows, in_cols) = img;
        return padded;
    }

public:
    MeanPooling(int patch, int stride, int padding, int in_rows, int in_cols)
        : patch(patch), stride(stride), padding(padding),
        in_rows(in_rows), in_cols(in_cols) {
        in_rows_pad = in_rows + 2 * padding;
        in_cols_pad = in_cols + 2 * padding;
        out_rows = (in_rows_pad - patch) / stride + 1;
        out_cols = (in_cols_pad - patch) / stride + 1;
    }

    tensor4d forward(const tensor4d& input) override {
        tensor4d output(input.size(), tensor3d(input[0].size(), mat(out_rows, out_cols)));
        for (size_t b = 0; b < input.size(); ++b) {
            for (size_t c = 0; c < input[0].size(); ++c) {
                mat fm_pad = apply_padding(input[b][c]);
                for (int i = 0; i < out_rows; ++i) {
                    for (int j = 0; j < out_cols; ++j) {
                        int row = i * stride;
                        int col = j * stride;
                        mat block = fm_pad.block(row, col, patch, patch);
                        output[b][c](i, j) = block.mean();
                    }
                }
            }
        }
        input_cache = input;
        return output;
    }

    tensor4d backward(const tensor4d& grad_output) override {
        tensor4d grad_input(input_cache.size(),
            tensor3d(input_cache[0].size(),
                mat::Zero(in_rows, in_cols)));
        double factor = 1.0 / (patch * patch);
        for (size_t b = 0; b < grad_output.size(); ++b) {
            for (size_t c = 0; c < grad_output[0].size(); ++c) {
                for (int i = 0; i < out_rows; ++i) {
                    for (int j = 0; j < out_cols; ++j) {
                        int row = i * stride;
                        int col = j * stride;
                        double grad_val = grad_output[b][c](i, j) * factor;
                        // Map back to original unpadded positions
                        int orig_row_start = row - padding;
                        int orig_col_start = col - padding;
                        for (int kr = 0; kr < patch; ++kr) {
                            for (int kc = 0; kc < patch; ++kc) {
                                int orig_r = orig_row_start + kr;
                                int orig_c = orig_col_start + kc;
                                if (orig_r >= 0 && orig_r < in_rows &&
                                    orig_c >= 0 && orig_c < in_cols) {
                                    grad_input[b][c](orig_r, orig_c) += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        return grad_input;
    }

    void update_params(double) override {}
    string name() const override { return "MeanPooling"; }
};

class Linear : public Layer {
private:
    mat weights;          // in_dim x out_dim
    rvec bias;            // out_dim
    tensor4d input_cache; // [batch][in_dim][1][1]
    mat grad_weights;
    rvec grad_bias;
    int in_dim, out_dim;

public:
    Linear(int in_dim, int out_dim)
        : in_dim(in_dim), out_dim(out_dim) {
        // He initialization
        weights = mat::Random(in_dim, out_dim) * sqrt(2.0 / in_dim);
        bias = rvec::Zero(out_dim);
        grad_weights = mat::Zero(in_dim, out_dim);
        grad_bias = rvec::Zero(out_dim);
    }

    tensor4d forward(const tensor4d& input) override {
        input_cache = input;
        int batch_size = input.size();
        tensor4d output(batch_size, tensor3d(out_dim, mat(1, 1)));
        for (int b = 0; b < batch_size; ++b) {
            // Flatten input: [in_dim] vector
            vec flat_in(in_dim);
            for (int i = 0; i < in_dim; ++i) {
                flat_in(i) = input[b][i](0, 0);
            }
            vec flat_out = weights.transpose() * flat_in + bias.transpose();
            for (int i = 0; i < out_dim; ++i) {
                output[b][i](0, 0) = flat_out(i);
            }
        }
        return output;
    }

    tensor4d backward(const tensor4d& grad_output) override {
        int batch_size = grad_output.size();
        grad_weights.setZero();
        grad_bias.setZero();
        tensor4d grad_input(batch_size, tensor3d(in_dim, mat(1, 1)));
        for (int b = 0; b < batch_size; ++b) {
            // Flatten grad_output
            vec flat_grad_out(out_dim);
            for (int i = 0; i < out_dim; ++i) {
                flat_grad_out(i) = grad_output[b][i](0, 0);
            }
            // Flatten input
            vec flat_in(in_dim);
            for (int i = 0; i < in_dim; ++i) {
                flat_in(i) = input_cache[b][i](0, 0);
            }
            // Accumulate gradients
            grad_weights += flat_in * flat_grad_out.transpose();
            grad_bias += flat_grad_out.transpose();
            // Input gradient
            vec flat_grad_in = weights * flat_grad_out;
            for (int i = 0; i < in_dim; ++i) {
                grad_input[b][i](0, 0) = flat_grad_in(i);
            }
        }
        return grad_input;
    }

    void update_params(double learning_rate) override {
        int batch_size = input_cache.size();
        double factor = learning_rate / batch_size;
        weights -= factor * grad_weights;
        bias -= factor * grad_bias;
    }

    string name() const override { return "Linear"; }
};

class Softmax : public Layer {
public:
    tensor4d forward(const tensor4d& input) override {
        tensor4d output(input.size(), tensor3d(input[0].size(), mat(1, 1)));
        for (size_t b = 0; b < input.size(); ++b) {
            // Find max for numerical stability
            double max_val = -numeric_limits<double>::infinity();
            for (size_t c = 0; c < input[0].size(); ++c) {
                max_val = max(max_val, input[b][c](0, 0));
            }
            double sum = 0.0;
            vector<double> exp_vals(input[0].size());
            for (size_t c = 0; c < input[0].size(); ++c) {
                exp_vals[c] = exp(input[b][c](0, 0) - max_val);
                sum += exp_vals[c];
            }
            for (size_t c = 0; c < input[0].size(); ++c) {
                output[b][c](0, 0) = exp_vals[c] / sum;
            }
        }
        return output;
    }

    // Note: backward not implemented here because gradient is combined with cross-entropy loss
    tensor4d backward(const tensor4d&) override {
        throw logic_error("Softmax backward should be handled by loss function");
    }
    void update_params(double) override {}
    string name() const override { return "Softmax"; }
};

class ResidualUnit : public Layer {
private:
    Conv conv1, conv2;
    ReLu relu1, relu2;

public:
    ResidualUnit(int patch, int stride, int padding, int num_filters,
        int channels, int input_rows, int input_cols)
        : conv1(patch, stride, padding, num_filters, input_rows, input_cols, channels),
        conv2(patch, stride, padding, num_filters, input_rows, input_cols, num_filters) {}

    tensor4d forward(const tensor4d& input) override {
        auto z1 = conv1.forward(input);
        auto a1 = relu1.forward(z1);
        auto z2 = conv2.forward(a1);
        // Skip connection: add input (need to match dimensions)
        // Here we assume input shape is same as z2 shape (which holds if stride=1, padding same)
        for (size_t b = 0; b < z2.size(); ++b) {
            for (size_t c = 0; c < z2[0].size(); ++c) {
                z2[b][c] += input[b][c];
            }
        }
        auto output = relu2.forward(z2);
        return output;
    }

    tensor4d backward(const tensor4d& grad_output) override {
        auto grad = relu2.backward(grad_output);
        grad = conv2.backward(grad);
        grad = relu1.backward(grad);
        // Gradient from skip connection
        auto grad_skip = grad;   // shape same as input
        grad = conv1.backward(grad);
        // Sum gradients from conv path and skip path
        for (size_t b = 0; b < grad.size(); ++b) {
            for (size_t c = 0; c < grad[0].size(); ++c) {
                grad[b][c] += grad_skip[b][c];
            }
        }
        return grad;
    }

    void update_params(double lr) override {
        conv1.update_params(lr);
        conv2.update_params(lr);
    }

    string name() const override { return "ResidualUnit"; }
};

class Flatten : public Layer {
private:
    vector<tuple<int, int, int>> shapes;
public:
    tensor4d forward(const tensor4d& input) override {
        shapes.clear();
        tensor4d output(input.size());
        for (size_t b = 0; b < input.size(); ++b) {
            int channels = input[b].size();
            int rows = input[b][0].rows();
            int cols = input[b][0].cols();
            shapes.push_back({ channels, rows, cols });
            int total = channels * rows * cols;
            tensor3d sample(total, mat(1, 1));
            int idx = 0;
            for (int c = 0; c < channels; ++c) {
                const mat& m = input[b][c];
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        sample[idx++](0, 0) = m(i, j);
                    }
                }
            }
            output[b] = sample;
        }
        return output;
    }

    tensor4d backward(const tensor4d& grad_output) override {
        tensor4d grad_input(grad_output.size());
        for (size_t b = 0; b < grad_output.size(); ++b) {
            auto [channels, rows, cols] = shapes[b];
            tensor3d sample(channels, mat(rows, cols));
            int idx = 0;
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        sample[c](i, j) = grad_output[b][idx++](0, 0);
                    }
                }
            }
            grad_input[b] = sample;
        }
        return grad_input;
    }
    void update_params(double) override {}
    string name() const override { return "Flatten"; }
};

class Simple_CNN {
private:
    vector<Layer*> layers;
    bool is_fitted;
    int classes;
public:
    //only support square and little photo like mnist
    //if want bigger photo or more complicated tasks
    //you should build CNN by yourself
    Simple_CNN(int photo_size, int channles):is_fitted(false) {
        layers.push_back(new Conv(3, 1, 1, 16, photo_size, photo_size, channles));
        layers.push_back(new ReLu());
        layers.push_back(new MaxPooling(2, 2, 0, photo_size, photo_size));
        layers.push_back(new Conv(3, 1, 1, 32, photo_size / 2, photo_size / 2, 16));
        layers.push_back(new ReLu());
        layers.push_back(new MaxPooling(2, 2, 0, photo_size / 2, photo_size / 2));
        layers.push_back(new Flatten());
        layers.push_back(new Linear(2 * photo_size * photo_size, 128));
        layers.push_back(new ReLu());
        layers.push_back(new Linear(128, 10));
        layers.push_back(new Softmax());
    }
    ~Simple_CNN() {
        for (auto layer : layers) delete layer;
    }
    void fit(const tensor4d& X_train,
        const vector<int>& y_train,
        int classes,
        int epoches = 10,
        double learning_rate = 0.001,
        int batch = 32,
        bool verbose = true) {
        is_fitted = true;
        this->classes = classes;
        int number_of_train = y_train.size();
        vector<int> y_seq(number_of_train);
        iota(y_seq.begin(), y_seq.end(), 0);
        random_device rd;
        auto seed = rd();
        mt19937 gen(seed);
        for (int epoch = 1; epoch <= epoches; epoch++) {
            shuffle(y_seq.begin(), y_seq.end(), gen);
            double total_loss = 0.0;
            for (int i = 0; i < number_of_train; i += batch) {
                int end = min(i + batch, number_of_train);
                int true_batch = end - i;
                tensor4d X_batch;
                X_batch.reserve(true_batch);
                for (int j = i; j < end; j++) X_batch.push_back(X_train[y_seq[j]]);
                vector<int> y_batch;
                y_batch.reserve(true_batch);
                for (int j = i; j < end; j++) y_batch.push_back(y_train[y_seq[j]]);
                for (auto layer : layers) {
                    X_batch = layer->forward(X_batch);
                }
                for (int b = 0; b < true_batch; b++) {
                    for (int c = 0; c < classes; c++) {
                        if (c == y_batch[b]) {
                            total_loss -= log(X_batch[b][c](0, 0) + 1e-8);
                            X_batch[b][c](0, 0) -= 1.0;
                        }
                    }
                }
                for (int layer = layers.size() - 2; layer >= 0; layer--) {
                    X_batch = layers[layer]->backward(X_batch);
                }
                for (auto layer : layers) layer->update_params(learning_rate);
            }
            if (verbose) {
                cout << "epoch " << epoch << " loss: " << total_loss / (double)number_of_train << endl;
            }
        }
    }
    vector<int> predict(const tensor4d& X_test) {
        vector<int> y_test;
        if (!is_fitted) return y_test;
        int number_of_test = X_test.size();
        auto X = X_test;
        for (int i = 0; i < layers.size() - 1; i++) X = layers[i]->forward(X);
        y_test.resize(number_of_test);
        for (int b = 0; b < number_of_test; b++) {
            double max_pro = -numeric_limits<double>::infinity();
            int max_index;
            for (int i = 0; i < classes; i++) {
                if (X[b][i](0, 0) > max_pro) {
                    max_pro = X[b][i](0, 0);
                    max_index = i;
                }
            }
            y_test[b] = max_index;
        }
        return y_test;
    }
};