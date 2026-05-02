#pragma once
#include "FNN.hpp"

class Gelu :public Layer {

};

class Softmax_Attention :public Layer {
private:
    mat output_cache;
public:
    mat forward(const mat& input) override {
        mat shifted = input.colwise() - input.rowwise().maxCoeff(); 
        mat exp_vals = shifted.array().exp();
        mat probabilities = exp_vals.array().colwise() / exp_vals.rowwise().sum().array();
        output_cache = probabilities;
        return probabilities;
    }
    mat backward(const mat& grad_output) override {
        mat grad_input(output_cache.rows(), output_cache.cols());
        for (int i = 0; i < output_cache.rows(); i++) {
            vec s = output_cache.row(i).transpose();
            mat jacobi = -s * s.transpose();
            jacobi.diagonal().array() += s.array();
            vec output_row = grad_output.row(i).transpose();
            grad_input.row(i) = (jacobi * output_row).transpose();
        }
        return grad_input;
    }
    string name() const override { return "Softmax_Attention"; }
};


class Self_Attention :public Layer {
    int embed_dim;
    int num_heads;
    int d_k;
    Linear W_O, W_Q, W_K, W_V;
    Softmax_Attention softmax;
    mat input_cache;
    mat Q_cache, K_cache, V_cache;
    mat scores_cache;
    mat attend_cache;
    mat head_out_cache;
public:
    Self_Attention(int embed_dim, int num_heads) :
        embed_dim(embed_dim), num_heads(num_heads), d_k(embed_dim / num_heads),
        W_Q(embed_dim, embed_dim), W_K(embed_dim, embed_dim),
        W_V(embed_dim, embed_dim), W_O(embed_dim, embed_dim) {}
    mat forward(const mat& input) override {
        input_cache = input;
        int seq_len = input.rows();
        Q_cache = W_Q.forward(input);
        K_cache = W_K.forward(input);
        V_cache = W_V.forward(input);
        mat output = mat::Zero(seq_len, embed_dim);
        attend_cache.resize(num_heads * seq_len, seq_len);
        for (int head = 0; head < num_heads; head++) {
            int start = head * d_k;
            mat Q_head = Q_cache.block(0, start, seq_len, d_k);
            mat K_head = K_cache.block(0, start, seq_len, d_k);
            mat V_head = V_cache.block(0, start, seq_len, d_k);
            mat scores = Q_head * K_head.transpose() / sqrt(d_k);
            mat attend = softmax.forward(scores);
            attend_cache.block(head * seq_len, 0, seq_len, seq_len) = attend;
            mat head_out = attend * V_head;
            output.block(0, start, seq_len, d_k) = head_out;
        }
        head_out_cache = output;
        output = W_O.forward(output);
        return output;
    }
    mat backward(const mat& grad_output) override {
        int seq_len = input_cache.rows();
        mat grad_head_out = W_O.backward(grad_output);
        mat grad_Q = mat::Zero(seq_len, embed_dim);
        mat grad_K = mat::Zero(seq_len, embed_dim);
        mat grad_V = mat::Zero(seq_len, embed_dim);
        for (int head = 0; head < num_heads; head++) {
            int start = head * d_k;
            mat grad_head = grad_head_out.block(0, start, seq_len, d_k);
            mat attend = attend_cache.block(head * seq_len, 0, seq_len, seq_len);
            mat Q_head = Q_cache.block(0, start, seq_len, d_k);
            mat K_head = K_cache.block(0, start, seq_len, d_k);
            mat V_head = V_cache.block(0, start, seq_len, d_k);
            mat grad_V_head = attend.transpose() * grad_head;
            grad_V.block(0, start, seq_len, d_k) += grad_V_head;
            mat grad_attend = grad_head * V_head.transpose();
            mat grad_scores = softmax.backward(grad_attend) / sqrt(d_k);
            mat grad_Q_head = grad_scores * K_head;
            mat grad_K_head = grad_scores.transpose() * Q_head;
            grad_Q.block(0, start, seq_len, d_k) += grad_Q_head;
            grad_K.block(0, start, seq_len, d_k) += grad_K_head;
        }
        mat grad_input = W_Q.backward(grad_Q) + W_K.backward(grad_K) + W_V.backward(grad_V);
        return grad_input;
    }
    void update_params(double learning_rate) override {
        W_Q.update_params(learning_rate);
        W_K.update_params(learning_rate);
        W_V.update_params(learning_rate);
        W_O.update_params(learning_rate);
    }
    string name() const override { return "Self_Attention"; }
};

class Layer_Norm :public Layer {
private:
    int embed_dim;
    rvec gamma;
    rvec beta;
    rvec grad_gamma;
    rvec grad_beta;
    mat input_cache;
    vec mean_cache;
    vec var_cache;
    mat normalized_cache;
    rvec m_gamma, v_gamma, m_beta, v_beta;
    double beta1, beta2, eps;
    int t;
public:
    Layer_Norm(int embed_dim) :embed_dim(embed_dim),
        beta1(0.9), beta2(0.999), eps(1e-8), t(0) {
        gamma = rvec::Ones(embed_dim);
        beta = rvec::Zero(embed_dim);
        grad_gamma = rvec::Zero(embed_dim);
        grad_beta = rvec::Zero(embed_dim);
        m_gamma = rvec::Zero(embed_dim);
        v_gamma = rvec::Zero(embed_dim);
        m_beta = rvec::Zero(embed_dim);
        v_beta = rvec::Zero(embed_dim);
    }
    mat forward(const mat& input) override {
        input_cache = input;
        int seq_len = input.rows();
        mat result(seq_len, embed_dim);
        mean_cache.resize(seq_len);
        var_cache.resize(seq_len);
        normalized_cache.resize(seq_len, embed_dim);
        for (int i = 0; i < seq_len; i++) {
            double mean = input.row(i).mean();
            double var = (input.row(i).array() - mean).square().mean();
            mean_cache(i) = mean;
            var_cache(i) = var;
            normalized_cache.row(i) = (input.row(i).array() - mean) / sqrt(var + 1e-6);
            result.row(i) = gamma.array() * normalized_cache.row(i).array() + beta.array();
        }
        return result;
    }
    mat backward(const mat& grad_output) override {
        int seq_len = input_cache.rows();
        mat grad_input(seq_len, embed_dim);
        for (int i = 0; i < seq_len; i++) {
            grad_beta += grad_output.row(i);
            grad_gamma.array() += (grad_output.row(i).array() * normalized_cache.row(i).array());
        }
        for (int i = 0; i < seq_len; i++) {
            rvec dy = grad_output.row(i);
            rvec x_center = input_cache.row(i).array() - mean_cache(i);
            double inv_std = 1.0 / sqrt(var_cache(i) + 1e-6);
            rvec gamma_dy = gamma.array() * dy.array();
            rvec term1 = gamma_dy.array() * inv_std;
            double sum_gamma_dy = gamma_dy.sum();
            rvec term2 = rvec::Constant(embed_dim, sum_gamma_dy * inv_std / (double)embed_dim);
            double sum_gamma_dy_center = (gamma_dy.array() * x_center.array()).sum();
            sum_gamma_dy_center *= pow(inv_std, 3) / (double)embed_dim;
            rvec term3 = x_center.array() * sum_gamma_dy_center;
            grad_input.row(i) = term1.array() - term2.array() - term3.array();
        }
        return grad_input;
    }
    void update_params(double learning_rate) override {
        double max_norm = 1.0;
        if (grad_gamma.norm() > max_norm)
            grad_gamma *= max_norm / grad_gamma.norm();
        if (grad_beta.norm() > max_norm)
            grad_beta *= max_norm / grad_beta.norm();
        t++;
        for (int j = 0; j < embed_dim; j++) {
            double g = grad_gamma(j);
            m_gamma(j) = beta1 * m_gamma(j) + (1 - beta1) * g;
            v_gamma(j) = beta2 * v_gamma(j) + (1 - beta2) * g * g;
            double m_hat = m_gamma(j) / (1 - pow(beta1, t));
            double v_hat = v_gamma(j) / (1 - pow(beta2, t));
            gamma(j) -= learning_rate * m_hat / (sqrt(v_hat) + eps);
            g = grad_beta(j);
            m_beta(j) = beta1 * m_beta(j) + (1 - beta1) * g;
            v_beta(j) = beta2 * v_beta(j) + (1 - beta2) * g * g;
            m_hat = m_beta(j) / (1 - pow(beta1, t));
            v_hat = v_beta(j) / (1 - pow(beta2, t));
            beta(j) -= learning_rate * m_hat / (sqrt(v_hat) + eps);
        }
        grad_gamma.setZero();
        grad_beta.setZero();
    }
    string name() const override { return "LayerNorm"; }
};

class FFN :public Layer {
private:
    Linear W1, W2;
    ReLU relu;
public:
    FFN(int embed_dim, int hiden_dim) :
        W1(embed_dim, hiden_dim), W2(hiden_dim, embed_dim) {}
    mat forward(const mat& input) override {
        mat hidden = W1.forward(input);
        hidden = relu.forward(hidden);
        hidden = W2.forward(hidden);
        return hidden;
    }
    mat backward(const mat& grad_output) override {
        mat grad_hidden = W2.backward(grad_output);
        grad_hidden = relu.backward(grad_hidden);
        grad_hidden = W1.backward(grad_hidden);
        return grad_hidden;
    }
    void update_params(double learning_rate) override {
        W1.update_params(learning_rate);
        W2.update_params(learning_rate);
    }
    string name() const override { return "FFN"; }
};

class Transfomer_Encoder_Layer :public Layer {
private:
    Layer_Norm norm1, norm2;
    Self_Attention self_attend;
    FFN ffn;
public:
    Transfomer_Encoder_Layer(int embed_dim, int num_heads, int hiden_dim) :
        norm1(embed_dim), norm2(embed_dim),
        self_attend(embed_dim, num_heads),
        ffn(embed_dim, hiden_dim) {}
    mat forward(const mat& input) override {
        mat norm1_out = norm1.forward(input);
        mat attend_out = self_attend.forward(norm1_out);
        mat add1 = attend_out + input;
        mat norm2_out = norm2.forward(add1);
        mat ffn_out = ffn.forward(norm2_out);
        mat add2 = add1 + ffn_out;
        return add2;
    }
    mat backward(const mat& grad_output) override {
        mat grad_ffn = ffn.backward(grad_output);
        mat grad_x1 = grad_output + norm2.backward(grad_ffn);
        mat grad_attend = self_attend.backward(grad_x1);
        mat grad_input = grad_x1 + norm1.backward(grad_attend);
        return grad_input;
    }
    void update_params(double learning_rate) override {
        norm1.update_params(learning_rate);
        norm2.update_params(learning_rate);
        ffn.update_params(learning_rate);
        self_attend.update_params(learning_rate);
    }
    string name() const override { return "Transfomer_Encoder_Layer"; }
};

class ave_pooling :public Layer {
    int seq_len;
public:
    mat forward(const mat& input) override {
        seq_len = input.rows();
        mat output = input.colwise().sum();
        output = output.array() / (double)seq_len;
        return output;
    }
    mat backward(const mat& grad_output) override {
        mat grad_input = grad_output.array() / (double)seq_len;
        grad_input = grad_input.replicate(seq_len, 1);
        return grad_input;
    }
    string name() const override {
        return "ave_pooling";
    }
};

class Transformer_Encoder_regression {
private:
    vector<Layer*> layers;
    bool is_fitted;
    int number_of_train;
public:
    Transformer_Encoder_regression(int num_layers, int embed_dim, int num_heads, int hiden_dim) {
        layers.push_back(new Linear(1, embed_dim));
        for (int i = 0; i < num_layers; i++) {
            layers.push_back(new Transfomer_Encoder_Layer(embed_dim, num_heads, hiden_dim));
        }
        layers.push_back(new ave_pooling());
        layers.push_back(new Linear(embed_dim, 1));
    }
    ~Transformer_Encoder_regression() {
        for (auto layer : layers) delete layer;
    }
    double one_step_train(const Point& X, const double y, double learning_rate) {
        int seq_len = X.size();
        vec in = Map<const vec>(X.data(), seq_len);
        mat input = in.reshaped(seq_len, 1);
        for (auto layer : layers) {
            input = layer->forward(input);
        }
        double diff = input.value() - y;
        double loss = diff * diff;
        mat grad(1, 1);
        grad(0, 0) = diff;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers[i]->backward(grad);
        }
        for (auto layer : layers) {
            layer->update_params(learning_rate);
        }
        return loss;
    }
    void fit(const vector<Point>& X_train,
        const vector<double>& y_train, int epoches = 50,
        double learning_rate = 0.1,
        bool verbose = true) {
        is_fitted = true;
        number_of_train = y_train.size();
        random_device rd;
        auto seed = rd();
        mt19937 gen(seed);
        vector<int> batch_seq(number_of_train);
        iota(batch_seq.begin(), batch_seq.end(), 0);
        for (int epoch = 1; epoch <= epoches; epoch++) {
            shuffle(batch_seq.begin(), batch_seq.end(), gen);
            double total_loss = 0;
            for (int i = 0; i < number_of_train; i++) {
                total_loss += one_step_train(X_train[batch_seq[i]], y_train[batch_seq[i]], learning_rate);
            }
            if (verbose) {
                cout << "epoch " << epoch << ", loss: " << total_loss / (double)number_of_train << '\n';
            }
        }
    }
    vector<double> predict(const vector<Point>& X_test) {
        vector<double> y_test;
        if (!is_fitted) return y_test;
        int number_of_test = X_test.size();
        y_test.resize(number_of_test);
        for (int i = 0; i < number_of_test; i++) {
            int seq_len = X_test[i].size();
            vec in = Map<const vec>(X_test[i].data(), seq_len);
            mat input = in.reshaped(seq_len, 1);
            for (auto layer : layers) {
                input = layer->forward(input);
            }
            y_test[i] = input.value();
        }
        return y_test;
    }
};

class Position_Encoding :public Layer {
private:
    mat pos;
    mat grad_loss;
    mat m_pos, v_pos;
    double beta1, beta2, eps;
    int t;
public:
    Position_Encoding(int seq_len, int embed_dim) :pos(seq_len, embed_dim),
        beta1(0.9), beta2(0.999), eps(1e-8), t(0) {
        pos.setRandom();
        pos *= 0.02;
        grad_loss = mat::Zero(seq_len, embed_dim);
        m_pos = mat::Zero(seq_len, embed_dim);
        v_pos = mat::Zero(seq_len, embed_dim);
    }
    mat forward(const mat& input) override {
        mat output = pos.block(0, 0, input.rows(), input.cols()) + input;
        return output;
    }
    mat backward(const mat& grad_output) override {
        grad_loss.block(0, 0, grad_output.rows(), grad_output.cols()) += grad_output;
        return grad_output;
    }
    void update_params(double learning_rate) override {
        double max_norm = 1.0;
        if (grad_loss.norm() > max_norm)
            grad_loss *= max_norm / grad_loss.norm();
        t++;
        for (int i = 0; i < grad_loss.rows(); i++) {
            for (int j = 0; j < grad_loss.cols(); j++) {
                double g = grad_loss(i, j);
                m_pos(i, j) = beta1 * m_pos(i, j) + (1 - beta1) * g;
                v_pos(i, j) = beta2 * v_pos(i, j) + (1 - beta2) * g * g;
                double m_hat = m_pos(i, j) / (1 - pow(beta1, t));
                double v_hat = v_pos(i, j) / (1 - pow(beta2, t));
                pos(i, j) -= learning_rate * m_hat / (sqrt(v_hat) + eps);
            }
        }
        grad_loss.setZero();
    }
    string name() const override { return "Position_Encoding"; }
};


class Transformer_Encoder_classify {
protected:
    vector<Layer*> layers;
    bool is_fitted;
    int number_of_train;
    int num_class;
    int batch_size;
    Transformer_Encoder_classify() = default;
public:
    Transformer_Encoder_classify(int num_layers, int embed_dim, int num_heads,
        int batch_size, int hiden_dim, int num_class,int max_seqlen) :
        num_class(num_class),batch_size(batch_size) {
        layers.push_back(new Linear(1, embed_dim));
        layers.push_back(new Position_Encoding(max_seqlen, embed_dim));
        for (int i = 0; i < num_layers; i++) {
            layers.push_back(new Transfomer_Encoder_Layer(embed_dim, num_heads, hiden_dim));
        }
        layers.push_back(new ave_pooling());
        layers.push_back(new Linear(embed_dim, num_class));
        layers.push_back(new Softmax_Attention());
    }
    ~Transformer_Encoder_classify() {
        for (auto layer : layers) delete layer;
    }
    double one_step_train(const Point& X, const Point& y_onehot,int true_batch, double learning_rate) {
        int seq_len = X.size();
        vec in = Map<const vec>(X.data(), seq_len);
        mat input = in.reshaped(seq_len, 1);

        for (auto layer : layers) {
            input = layer->forward(input);
        }

        rvec pred = input.row(0);  // [1, num_class]
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
    vector<Point> to_onehot(const vector<int>& y_train) const {
        vector<Point> y_onehot(number_of_train, Point(num_class, 0.0));
        for (int i = 0; i < number_of_train; i++) {
            y_onehot[i][y_train[i]] = 1.0;
        }
        return y_onehot;
    }
    void fit(const vector<Point>& X_train,
        const vector<int>& y_train,
        int epoches = 50,
        double learning_rate = 0.01,
        bool verbose = true) {
        is_fitted = true;
        number_of_train = y_train.size();
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
                    total_loss += one_step_train(X_train[batch_seq[j]], y_onehot[batch_seq[j]], true_batch, learning_rate);
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
            int seq_len = X_test[i].size();
            vec in = Map<const vec>(X_test[i].data(), seq_len);
            mat input = in.reshaped(seq_len, 1);
            for (int j = 0; j < layers.size() - 1; j++) {
                input = layers[j]->forward(input);
            }
            input.row(0).maxCoeff(&y_test[i]);
        }
        return y_test;
    }
};
