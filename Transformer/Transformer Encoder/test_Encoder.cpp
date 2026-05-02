#include <iostream>
#include "Vision_Transformer.hpp"
#include <fstream>
#include <sstream>

const int photo_size = 28;
const int patch_size = 4;
constexpr int seq_len = photo_size * photo_size;

int main() {
	ifstream image("mnist_txt/train_images.txt");
	ifstream label("mnist_txt/train_labels.txt");
	string line;
	vector<Point> X_train;
	X_train.reserve(60000);
	while (getline(image, line)) {
		Point X; X.reserve(seq_len);
		istringstream iss(line);
		double x;
		while (iss >> x) X.push_back(x);
		X_train.push_back(move(X));
	}
	vector<int> y_train;
	y_train.reserve(X_train.size());
	int y;
	while (label >> y) y_train.push_back(y);
	Vision_transformer_Encoder encoder(2, 64, 2, 256, 128, 10, patch_size, photo_size);
	encoder.fit(X_train, y_train, 20, 0.0001);
	ifstream test_image("mnist_txt/test_images.txt");
	ifstream test_label("mnist_txt/test_labels.txt");
	X_train.clear();
	y_train.clear();
	while (getline(test_image, line)) {
		Point X; X.reserve(seq_len);
		istringstream iss(line);
		double x;
		while (iss >> x) X.push_back(x);
		X_train.push_back(move(X));
	}
	while (test_label >> y) y_train.push_back(y);
	cout << "Test " << y_train.size() << " accuracy: " << encoder.accuracy(X_train, y_train) << endl;
}