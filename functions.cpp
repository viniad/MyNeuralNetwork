#include "Matrix.hpp"
Matrix* fromArray(vector<double> arr){
    Matrix* m = new Matrix(arr.size(), 1);

    for(int i = 0; i < arr.size(); i++){
        m->data[i][0] = arr[i];
    }
    return m;
}

Matrix* multiply(Matrix* m, double b){
    Matrix* result = new Matrix(m->getRows(), m->getCols());
    int i, j, k = 0;
    for(int i = 0; i<m->getRows(); i++){
        for(int j = 0; j<m->getCols(); j++){
            m->data[i][j] *= b;
        }
    }
    return result;
}

//Matrix product
Matrix* multiply(Matrix* a, Matrix* b){
    if(a->getCols() != b->getRows()){
        cout << "(Mul) Columns of A must match Rows of B" << endl;
    }
    int i, j, k = 0;

    Matrix *result = new Matrix(a->getRows(), b->getCols());
    result->zeros();

    for (i = 0; i < a->getRows(); i++) {
        for (j = 0; j < b->getCols(); j++) {
            for (k = 0; k < a->getCols(); k++) {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    return result;
}


Matrix* subtract(Matrix* a, Matrix* b){
    Matrix* result = new Matrix(a->getRows(), a->getCols());

    for(int i = 0; i < result->getRows(); i++){
        for(int j = 0; j < result->getCols(); j++){
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    return result;
}

Matrix* map(Matrix* a, string functionName){
    Matrix* result = new Matrix(a->getRows(), a->getCols());
    if (functionName == "sigmoid") {
        for (int i = 0; i < a->getRows(); i++) {
            for (int j = 0; j < a->getCols(); j++) {
                result->data[i][j] = sigmoid(a->data[i][j]);
            }
        }
    } else if (functionName == "dsigmoid") {
        for (int i = 0; i < a->getRows(); i++) {
            for (int j = 0; j < a->getCols(); j++) {
                result->data[i][j] = dsigmoid(a->data[i][j]);
            }
        }
    }
    return result;
}

Matrix* transpose(Matrix* m){
    if(m->getRows() == 1 && m->getCols() == 2){
        Matrix* result = new Matrix(2, 1);
        for(int i = 0; i < m->getRows(); i++){
            for(int j = 0; j < m->getCols(); j++){
                result->data[j][i] = m->data[i][j];
            }
        }
        return result;

    }else if(m->getRows() == 2 && m->getCols() == 1) {
        Matrix* result = new Matrix(1, 2);
        for(int i = 0; i < m->getRows(); i++){
            for(int j = 0; j < m->getCols(); j++){
                result->data[j][i] = m->data[i][j];
            }
        }
        return result;
    }else {
        Matrix *result = new Matrix(m->getRows(), m->getCols());
        for (int i = 0; i < m->getRows(); i++) {
            for (int j = 0; j < m->getCols(); j++) {
                result->data[j][i] = m->data[i][j];
            }
        }
        return result;
    }

}



