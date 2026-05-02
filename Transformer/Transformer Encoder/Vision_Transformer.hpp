#pragma once
#include "Transformer_Encoder.hpp"

class Patch_Embedding : public Layer {
private:
    int embed_dim;
    int patch_size;
    int photo_size;
    int seq_len;
    Linear projection;
public:
    Patch_Embedding(int patch_size, int photo_size, int embed_dim)
        : patch_size(patch_size), photo_size(photo_size), embed_dim(embed_dim),
        projection(patch_size* patch_size, embed_dim)
    {
        seq_len = (photo_size / patch_size) * (photo_size / patch_size);
    }

    mat forward(const mat& input) override {
        int patches_per_row = photo_size / patch_size;
        mat patches(seq_len, patch_size * patch_size);  

        for (int now = 0; now < seq_len; now++) {
            int row_start = (now / patches_per_row) * patch_size;
            int col_start = (now % patches_per_row) * patch_size;
            int iter = 0;
            for (int i = row_start; i < row_start + patch_size; i++) {
                for (int j = col_start; j < col_start + patch_size; j++) {
                    patches(now, iter++) = input(i, j);
                }
            }
        }

        return projection.forward(patches);  
    }

    mat backward(const mat& grad_output) override {
        projection.backward(grad_output);  
        return mat();
    }

    void update_params(double learning_rate) override {
        projection.update_params(learning_rate);
    }

    string name() const override { return "Patch_Embedding"; }
};

class Vision_transformer_Encoder :public Transformer_Encoder_classify {
private:
    int seq_len;
    int photo_size;
public:
    Vision_transformer_Encoder(int num_layers, int embed_dim, int num_heads,
        int batch_size, int hiden_dim, int num_class, int patch_size, int photo_size) :
        photo_size(photo_size),
        seq_len((photo_size / patch_size)* (photo_size / patch_size)) {
        this->num_class = num_class;
        this->batch_size = batch_size;
        layers.push_back(new Patch_Embedding(patch_size, photo_size, embed_dim));
        layers.push_back(new Position_Encoding(seq_len, embed_dim));
        for (int i = 0; i < num_layers; i++) {
            layers.push_back(new Transfomer_Encoder_Layer(embed_dim, num_heads, hiden_dim));
        }
        layers.push_back(new ave_pooling());
        layers.push_back(new Linear(embed_dim, num_class));
        layers.push_back(new Softmax_Attention());
    }
    double one_step_train(mat X, const vector<double>& y_onehot, int true_batch, double learning_rate) {
        for (auto layer : layers) {
            X = layer->forward(X);
        }
        rvec pred = X.row(0);  // [1, num_class]
        rvec y_vec = Map<const rvec>(y_onehot.data(), num_class);  // [1, num_class]

        double eps = 1e-12;
        double loss = -(y_vec.array() * (pred.array() + eps).log()).sum();

        mat grad_loss = pred - y_vec;  // [1, num_class]
        grad_loss /= (double)true_batch;
        for (int i = layers.size() - 2; i >= 0; i--) {//jump softmax
            grad_loss = layers[i]->backward(grad_loss);
        }
        return loss;
    }
    void fit(const vector<Point>& X_train,
        const vector<int>& y_train, int epoches = 50,
        double learning_rate = 0.01,
        bool verbose = true) {
        is_fitted = true;
        number_of_train = y_train.size();
        vector<mat> X_mat(number_of_train);
        for (int i = 0; i < number_of_train; i++) {
            X_mat[i] = Map<const mat>(X_train[i].data(), photo_size, photo_size);//persume photos are same square
        }
        vector<Point> y_onehot = move(to_onehot(y_train));
        random_device rd;
        auto seed = rd();
        mt19937 gen(seed);
        vector<int> batch_seq(number_of_train);
        iota(batch_seq.begin(), batch_seq.end(), 0);
        for (int epoch = 1; epoch <= epoches; epoch++) {
            shuffle(batch_seq.begin(), batch_seq.end(), gen);
            double total_loss = 0;
            for (int i = 0; i < number_of_train; i += batch_size) {
                int end = min(i + batch_size, number_of_train);
                int true_batch = end - i;
                for (int j = i; j < end; j++) {
                    total_loss += one_step_train(X_mat[batch_seq[j]], y_onehot[batch_seq[j]], true_batch, learning_rate);
                }
                for (auto layer : layers) {
                    layer->update_params(learning_rate);
                }
            }
            if (verbose) {
                cout << "epoch " << epoch << ", loss: " << total_loss / (double)number_of_train << '\n';
            }
        }
    }
    vector<int> predict(const vector<Point>& X_test) {
        vector<int> y_test;
        if (!is_fitted) return y_test;
        int number_of_test = X_test.size();
        y_test.resize(number_of_test);
        for (int i = 0; i < number_of_test; i++) {
            mat input = Map<const mat>(X_test[i].data(), photo_size, photo_size);//do not want to make size check
            for (int j = 0; j < layers.size() - 1; j++) {
                input = layers[j]->forward(input);
            }
            input.row(0).maxCoeff(&y_test[i]);
        }
        return y_test;
    }
    double accuracy(const vector<Point>& X_test, const vector<int>& y_test) {
        int number_of_test = y_test.size();
        int correct = 0;
        for (int i = 0; i < number_of_test; i++) {
            mat X = Map<const mat>(X_test[i].data(), photo_size, photo_size);
            for (int j = 0; j < layers.size() - 1; j++) {
                X = layers[j]->forward(X);
            }
            int idx;
            X.row(0).maxCoeff(&idx);
            if (idx == y_test[i]) correct++;
        }
        return (double)correct / (double)number_of_test;
    }
};