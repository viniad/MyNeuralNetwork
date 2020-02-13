#ifndef MYMLPNEURALNETWORK_MATRIX_HPP
#define MYMLPNEURALNETWORK_MATRIX_HPP

#include <cmath>
#include <cstdlib>

double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

//derivative of sigmoid
double dsigmoid(double y){
    //return sigmoid(x) * (1 - sigmoid(x));
    return y * (1 - y);
}

using namespace std;

class Matrix{
private:
    int rows;
    int cols;
public:
    double** data = nullptr;

    Matrix(int rows, int cols){
        this->rows = rows;
        this->cols = cols;
        data = createMatrix(rows, cols);
    }

    double** createMatrix(int r, int c){
        int i = 0;
        double** temp = (double**) malloc(sizeof(double*) * r);
        for(i = 0; i < r; i++){
            temp[i] = (double*) malloc(sizeof(double) * c);
        }
        return temp;
    }

    void print(){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                cout << this->data[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    void add(Matrix* n){
        if(this->rows != n->rows || this->cols != n->cols){
            cout << ("(Add) Columns and Rows of A must match Columns and Rows of B") << endl;
            return;
        }
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                this->data[i][j] += n->data[i][j];
            }
        }
    }


    Matrix* transpose(){
        if(this->rows == 1 && this->cols == 2){
            Matrix* result = new Matrix(2, 1);
            for(int i = 0; i < this->rows; i++){
                for(int j = 0; j < this->cols; j++){
                    result->data[j][i] = this->data[i][j];
                }
            }
            return result;
        }else if(this->rows == 2 && this->cols == 1) {
            Matrix* result = new Matrix(1, 2);
            for(int i = 0; i < this->rows; i++){
                for(int j = 0; j < this->cols; j++){
                    result->data[j][i] = this->data[i][j];
                }
            }
            return result;
        }else {
            Matrix *result = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result->data[j][i] = this->data[i][j];
                }
            }
            return result;
        }
    }

    //Hadamard product
    void multiply(Matrix* b){
        int i, j, k = 0;
        if(b->rows != 1 && b->cols != 1){
            if (this->rows != b->rows || this->cols != b->cols) {
                cout << "This is Hadamard product, Columns and Rows of A must match Columns and Rows of B." << endl;
                return;
            }
            //Hadamard product
            for(int i = 0; i<this->rows; i++){
                for(int j = 0; j<this->cols; j++){
                    this->data[i][j] *= b->data[i][j];
                }
            }
        }else if(b->rows == 1 && b->cols == 1){
            //Scalar product
            double value = b->data[0][0];
            this->multiply(value);
        }
    }

    //Scalar product
    void multiply(double b){
        int i, j, k = 0;
        for(int i = 0; i<this->rows; i++){
            for(int j = 0; j<this->cols; j++){
                this->data[i][j] *= b;
            }
        }
    }

    void randomize() {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] = rand() % 2;
            }
        }
    }


    Matrix* subtract(Matrix* b){
        Matrix* result = new Matrix(rows, cols);

        for(int i = 0; i < result->rows; i++){
            for(int j = 0; j < result->cols; j++){
                result->data[i][j] = this->data[i][j] - b->data[i][j];
            }
        }
        return result;
    }

    //Apply activation function to every element of matrix
    void map(string functionName) {
        if (functionName == "sigmoid") {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    this->data[i][j] = sigmoid(this->data[i][j]);
                }
            }
        } else if (functionName == "dsigmoid") {
            for (int i = 0; i < this->rows; i++) {
                for (int j = 0; j < this->cols; j++) {
                    this->data[i][j] = dsigmoid(this->data[i][j]);
                }
            }
        }
    }

    //Matrix to vector
    vector<double> toArray(){
        vector<double> vec;
        for(int i = 0; i<this->rows; i++){
            for(int j = 0; j<this->cols; j++) {
                vec.push_back(this->data[i][j]);
            }
        }
        return vec;
    }

    void zeros(){
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] = 0;
            }
        }
    }
    int getRows(){
        return this->rows;
    }

    int getCols(){
        return this->cols;
    }

    ~Matrix(){
        for(int i = 0; i < rows; i++){
            free(data[i]);
        }
        free(data);
    }
};




#endif //MYMLPNEURALNETWORK_MATRIX_HPP
