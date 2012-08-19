//Neural Network Training Code by Ramon Sandoval

#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <fstream>

using namespace std;

//Neural network class
class neuNet {

	public:
	
	vector<double> input; //values for the input layer
	vector<double> output; //values for the output layer
	vector<double> hidden; //values for the hidden layer
	
	//Vector of vectors holding edge weights. Outer vector corresponds to hidden nodes,
	//inner vector corresponds to input nodes
	vector< vector<double> > hid_in;

	//Vector of vectors holding edge weights. Outer vector corresponds to output nodes,
	//inner vector corresponds to hidden nodes	
	vector< vector<double> > out_hid;
	
	//Signoid function
	double g(double in) {
		return 1/(1+exp(-in));
	}
	
	//Signoid prime function
	double gp(double in) {
		return g(in)*(1-g(in));
	}
	
	double learningrate; //learning rate
	int epochs;	//number of epochs
};

//Class for holding example data
class in_out {
	public:
	
	vector<double> x; //input data
	vector<double> y; //output data
};

//Function for initializing neural network
void startNeural(neuNet &network) {
	string dummy;

	while(1) {
		cout << "Enter name of neural network: ";
		cin >> dummy;

		ifstream input;
		input.open(dummy.c_str());
		if(!input.is_open()) {
			cerr << "Sorry, but your file is in another castle\n";
			continue;
		}
		
		double size1, size2, size3;
		input >> size1; 
		input >> size2; 
		input >> size3;
		
		//Set number of inputs,hidden nodes, and outputs
		network.input.resize(size1);
		network.hidden.resize(size2);
		network.output.resize(size3);
		
		//Set size of outer vector of edges
		network.hid_in.resize(size2);
		network.out_hid.resize(size3);
		
		
		//Initialize input to hidden node edges
		for(int i = 0; i < size2; ++i) {
			network.hid_in[i].resize(size1+1); //size of edges must be 1 greater for bias weight
			for (int j = 0; j < size1+1; ++j) {
				input >> network.hid_in[i][j];
			}
		}
		
		//Initialize hidden node to output edges
		for(int i = 0; i < size3; ++i) {
			network.out_hid[i].resize(size2+1);
			for (int j = 0; j < size2+1; ++j) {
				input >> network.out_hid[i][j];
			}
		}
		
		input.close();
		return;
	}
}

//Function for initalizing example data
void startTraining(vector<in_out> &examples) {
	string dummy;

	while(1) {
		cout << "Enter name of training set: ";
		cin >> dummy;

		ifstream input;
		input.open(dummy.c_str());
		if(!input.is_open()) {
			cerr << "Sorry, but your file is in another castle\n";
			continue;
		}
		
		double size1, size2, size3;
		input >> size1; 
		input >> size2; 
		input >> size3;
		
		//Set number of examples
		examples.resize(size1);
		
		//Initialize example values
		for(int i = 0; i < size1; ++i) {
			examples[i].x.resize(size2);
			examples[i].y.resize(size3);
			for (int j = 0; j < size2; ++j) {
				input >> examples[i].x[j];
			}
			for (int j = 0; j < size3; ++j) {
				input >> examples[i].y[j];
			}
		}

		input.close();
		return;
	}
}

//Enter output file, epoch number, and learning rate
void startElse(neuNet &network, ofstream &output) {
	string dummy;

	//Enter output
	cout << "Enter name of output file: ";
	cin >> dummy;

	output.open(dummy.c_str());
	
	//Enter epoch number
	cout << "Enter number of epochs: ";
	cin >> network.epochs;
	
	//Enter learning rate
	cout << "Enter learning rate: ";
	cin >> network.learningrate;
}

//Find weighted sum using two vectors
double weightedsum(vector<double> weights, vector<double> a) {
	double c = -weights[0]; //corrsponds to bias weight
	
	//Find weighted sum of weights*value
	for (int i = 0; i < a.size(); ++i) {
		c += weights[i+1]*a[i];
	}
	return c;
}

//Find weighted sum between 1-d vector b and 2-d vector a, with
//a's second dimension locked in index to k
double anothersum(vector< vector<double> > a, vector<double> b, int k) {
	double c = 0;
	for (int i = 0; i < b.size(); ++i) {
		c += a[i][k]*b[i];
	}
	return c;
}

//Train neural network with back propogation learning
neuNet backPropLearning(vector<in_out> examples, neuNet network) {
	vector<double> deltaout(network.output.size()); //delta erros in output
	vector<double> deltahid(network.hidden.size()); //delta errors in hidden nodes
	
	//Loop until number of epochs is finished
	for(int epoch = 0; epoch < network.epochs; ++epoch ) {
		
		//Loop through example data
		for (int i = 0; i < examples.size(); ++i) {
			
			/*Propogate inputs forward*/
			
			//Initialize input values
			for (int nodes = 0; nodes < network.input.size(); ++nodes) {
				network.input[nodes] = examples[i].x[nodes];
			}
			
			//Find values of hidden nodes from inputs
			for (int nodes = 0; nodes < network.hidden.size(); ++nodes) {
				double in = weightedsum(network.hid_in[nodes], network.input);
				network.hidden[nodes] = network.g(in);
			}
			
			//Find values of outputs from hidden nodes
			for (int nodes = 0; nodes < network.output.size(); ++nodes) {
				double in = weightedsum(network.out_hid[nodes], network.hidden);
				network.output[nodes] = network.g(in);
			}
			
			/*Propogate deltas backward*/
			
			//Find delta errors of outputs
			for (int nodes = 0; nodes < network.output.size(); ++nodes) {
				double in = weightedsum(network.out_hid[nodes], network.hidden);
				deltaout[nodes] = network.gp(in)*(examples[i].y[nodes]-network.output[nodes]);
			}
			
			//Find delta errors of hidden nodes
			for (int nodes = 0; nodes < network.hidden.size(); ++nodes) {
				double in = weightedsum(network.hid_in[nodes], network.input);
				deltahid[nodes] = network.gp(in)*anothersum(network.out_hid,deltaout,nodes+1);
			}
			
			/*Update Weights*/
			
			//Update weights of hidden nodes to output edges
			for (int i = 0; i < network.out_hid.size(); ++i) {
				network.out_hid[i][0] += -network.learningrate*deltaout[i];
				for (int j = 1; j < network.out_hid[i].size(); ++j) {
					network.out_hid[i][j] += network.learningrate*network.hidden[j-1]*deltaout[i];
				}
			}
			
			//Update weights of inputs to hidden node edges
			for (int i = 0; i < network.hid_in.size(); ++i) {
				network.hid_in[i][0] += -network.learningrate*deltahid[i];
				for (int j = 1; j < network.hid_in[i].size(); ++j) {
					network.hid_in[i][j] += network.learningrate*network.input[j-1]*deltahid[i];
				}
			}
		}
	
	}
	
	return network;
}

//Output values for neural network
void outputNet(neuNet network, ofstream &output) {
	output.setf(ios::fixed,ios::floatfield);
	output.precision(3);
	
	output << network.input.size() << " "; //output number of inputs
	output << network.hidden.size() << " "; //output number of hidden nodes
	output << network.output.size() << endl; //output number of outputs
	
	//output weights of input-hidden node edges
	for (int i = 0; i < network.hid_in.size(); ++i) {
		for (int j = 0; j < network.hid_in[i].size(); ++j) {
			//if-else to prevent placing space at end of line
			if (j != network.hid_in[i].size()-1)
				output << network.hid_in[i][j] << " ";
			else
				output << network.hid_in[i][j];
		}
		output << endl;
	}
	
	//output weights of hidden node-output edges
	for (int i = 0; i < network.out_hid.size(); ++i) {
		for (int j = 0; j < network.out_hid[i].size(); ++j) {
			//if-else to prevent placing space at end of line
			if (j != network.out_hid[i].size()-1)
				output << network.out_hid[i][j] << " ";
			else
				output << network.out_hid[i][j];
		}
		output << endl;
	}
}

/*
void outputTrain(vector<in_out> examples, ofstream &output) {
	output.setf(ios::fixed,ios::floatfield);
	output.precision(3);
	
	output << examples.size() << " ";
	output << examples[0].x.size() << " ";
	output << examples[0].y.size() << endl;
	
	for (int i = 0; i < examples.size(); ++i) {
		output.setf(ios::fixed,ios::floatfield);
		output.precision(3);
		
		for (int j = 0; j < examples[i].x.size(); ++j) {
			output << examples[i].x[j] << " ";
		}
		
		output.setf(ios::fixed,ios::floatfield);
		output.precision(0);
		for (int j = 0; j < examples[i].y.size(); ++j) {
			if (j != examples[i].y.size()-1)
				output << examples[i].y[j] << " ";
			else
				output << examples[i].y[j];
		}
		output << endl;
	}
}*/

int main() {
	neuNet network; //neural network
	vector<in_out> examples; //training data
	ofstream output; //output stream
	
	startNeural(network); //Initialize neural network
	startTraining(examples); //Initialize training data
	startElse(network, output); //Open output file, initialize epoch number and learning rate
	
	//Train neural network
	outputNet(backPropLearning(examples, network),output);
}
