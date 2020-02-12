#ifndef MYMLPNEURALNETWORK_NEURALNETWORK_HPP
#define MYMLPNEURALNETWORK_NEURALNETWORK_HPP

#include "functions.cpp"
#include <fstream>

class NeuralNetwork{
private:
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    //int epochs = 0;

    //IH -> input to hidden
    //HO -> hidden to output
    Matrix* weights_ih;
    Matrix* weights_ho;
    Matrix* bias_h;
    Matrix* bias_o;
    double learningRate;

public:
    NeuralNetwork(int iNodes, int hNodes, int oNodes){
        this->inputNodes = iNodes;
        this->hiddenNodes = hNodes;
        this->outputNodes = oNodes;

        //IH -> input to hidden
        //HO -> hidden to output
        this->weights_ih = new Matrix(hNodes, iNodes);
        this->weights_ho = new Matrix(oNodes, hNodes);

        //H -> bias of hidden layer
        //O -> bias of output layer
        this->bias_h = new Matrix(this->hiddenNodes, 1);
        this->bias_o = new Matrix(this->outputNodes, 1);

        //Initializing the weights and biases randomly
        this->bias_h->randomize();
        this->bias_o->randomize();
        this->weights_ih->randomize();
        this->weights_ho->randomize();
        this->learningRate = 0.1;
    }

    ~NeuralNetwork(){
    };

    vector<double> predict(vector<double> input){

        //Generating the Hidden Outputs
        //H = I(i) * W(i,j) + B(i)
        Matrix* inputs = fromArray(input); //transforming an array into a matrix
        Matrix* hidden = multiply(this->weights_ih, inputs);
       // cout << "hidden" << endl;
        //hidden->print();
        //cout << "bias h" << endl;
        //this->bias_h->print();
        hidden->add(this->bias_h);

        //Activation function
        hidden->map("sigmoid");

        //Generating the output's Outputs
        Matrix* output = multiply(this->weights_ho, hidden);
        output->add(this->bias_o);
        output->map("sigmoid");

        return output->toArray();
    }


    void train(vector<double> inputArray, vector<double> targets /*answers*/, int epochs){
        for(int i = 0; i< epochs; i++){
            /////////cout << "Training......." << endl;
            //----Do the feedfoward algorithm
            //Generating the Hidden Outputs
            //H = I(i) * W(i,j) + B(i)
            Matrix* inputs = fromArray(inputArray);
            Matrix* hidden = multiply(this->weights_ih, inputs);
            hidden->add(this->bias_h);

            //Activation function
            hidden->map("sigmoid");

            //Generating the output's Outputs
            Matrix* outputs = multiply(this->weights_ho, hidden);
            outputs->add(this->bias_o);
            outputs->map("sigmoid");

            //----Training part
            //Convert array to matrix
            Matrix* targetsMatrix = fromArray(targets);

            //Output error = target - output
            //Matrix* outputErrors = targetsMatrix->subtract(outputs);
            Matrix* outputErrors;
            outputErrors = subtract(targetsMatrix, outputs);

            //////////Calculating the gradient of output layer
            ////////// W[i][j] = learningRate * outputerror * (output * (1-output)) * H(transpost)
            Matrix* outputGradients = map(outputs, "dsigmoid");
            outputGradients->multiply(outputErrors);
            outputGradients->multiply(this->learningRate);

            //Calculating hidden->output deltas
            Matrix* hiddenTransposed = transpose(hidden);
            Matrix* deltasWeights_ho = multiply(outputGradients, hiddenTransposed);
            this->weights_ho->add(deltasWeights_ho);
            //Adjusting the biases by its deltas (which is just the gradients)
            this->bias_o->add(outputGradients);

            //Hidden layer errors = weights matrix transpost * outputs errors matrix
            Matrix* weights_ho_T;
            weights_ho_T = transpose(weights_ho);
            Matrix* hiddenErrors = multiply(weights_ho_T, outputErrors);

            //Calculating the gradient of hidden layer
            // W[i][j] = learningRate * outputerror * (hidden * (1-hidden)) * I(transpost)
            Matrix* hiddenGradients = map(hidden, "dsigmoid");
            hiddenGradients->multiply(hiddenErrors);
            hiddenGradients->multiply(this->learningRate);

            //Calculating input->hidden deltas
            Matrix* inputsTransposed = transpose(inputs);
            Matrix* deltasWeights_ih = multiply(hiddenGradients, inputsTransposed);


            //Adjusting the hidden->output weights matrix
            this->weights_ih->add(deltasWeights_ih);
            // Adjust the bias by its deltas (which is just the gradients)
            this->bias_h->add(hiddenGradients);
        }
    }

    void save(string name){
        ofstream file;
        file.open(name + ".txt");
        file << "inputNodes:" << endl;
        file << this->inputNodes << endl;

        file << "hiddenNodes:" << endl;
        file << this->hiddenNodes << endl;

        file << "outputNodes:" << endl;
        file << this->outputNodes << endl;

        file << "WeightsIH:" << endl;
        for(int i = 0; i < this->weights_ih->getRows(); i++){
            for(int j = 0; j < this->weights_ih->getCols(); j++){
                file << this->weights_ih->data[i][j] << " ";
            }
            file << endl;
        }
        //file << endl;

        file << "WeightsHO:" << endl;
        for(int i = 0; i < this->weights_ho->getRows(); i++){
            for(int j = 0; j < this->weights_ho->getCols(); j++){
                file << this->weights_ho->data[i][j] << " ";
            }
            file << endl;
        }
        //file << endl;

        file << "BiasH:" << endl;
        for(int i = 0; i < this->bias_h->getRows(); i++){
            for(int j = 0; j < this->bias_h->getCols(); j++){
                file << this->bias_h->data[i][j] << " ";
            }
            file << endl;
        }
        //file << endl;

        file << "BiasO:" << endl;
        for(int i = 0; i < this->bias_o->getRows(); i++){
            for(int j = 0; j < this->bias_o->getCols(); j++){
                file << this->bias_o->data[i][j] << " ";
            }
            file << endl;
        }
        //file << endl;

        file << "LearningRate:" << endl;
        file << this->learningRate << endl;
        file.close();
    }


};


#endif //MYMLPNEURALNETWORK_NEURALNETWORK_HPP
