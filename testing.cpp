//Neural Network Testing Code by Ramon Sandoval

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
void startTesting(vector<in_out> &examples) {
	string dummy;

	while(1) {
		cout << "Enter name of testing set: ";
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

//Enter output file
void startElse(ofstream &output) {
	string dummy;

	//Enter output
	cout << "Enter name of output file: ";
	cin >> dummy;

	output.open(dummy.c_str());
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

//Find sum of all values within vector
double sum(vector<double> a) {
	double c = 0;
	for (int i = 0; i < a.size(); ++i) {
		c += a[i];
	}
	return c;
}

//Find average in 2-dimensional vector, output as 1-dimensional vector
vector<double> average(vector< vector<double> > a) {
	vector<double> c(a[0].size(),0);
	
	//Take sum of values across 2nd dimension of vectors
	for (int i = 0; i < a.size(); ++i) {
		for (int j = 0; j < a[i].size(); ++j) {
			c[j] += a[i][j];
		}
	}
	
	//Get average by dividing sums in each value by size of 1st dimension
	for (int j = 0; j < c.size(); ++j) {
		c[j] /= a.size();
	}
	
	return c;
}

//Find accuracy, precision, recall, and F1 all in one vector
vector<double> metrics(double A, double B, double C, double D) {
	vector<double> result(4,0);
	
	result[0] = (A + D)/(A + B + C + D); //accuracy
	result[1] = A / (A + B); //precision
	result[2] = A / (A + C); //recall
	result[3] = (2 * result[1] * result[2]) / (result[1] + result[2]); //F1
	
	return result;
}

//Find macro-averaged result
vector<double> macro(vector<double> A, vector<double> B, vector<double> C, vector<double> D) {
	vector< vector<double> > result1(A.size());
	vector<double> result;
	
	//Find vector of metrics, each cell of the vector
	//holding value of one metric
	for(int i = 0; i < A.size(); ++i) {
		result1[i] = metrics(A[i],B[i],C[i],D[i]);
	}
	
	//Find average among each of the metrics
	result = average(result1);
	
	//F1 has to be recalcuated using averge of precision and recall
	result[3] = (2 * result[1] * result[2]) / (result[1] + result[2]);
	
	return result;
}

//Use neural network and testing data to find calculate metrics
void test(vector<in_out> examples, neuNet network, ofstream &output) {
	vector<double> A(network.output.size(),0); //Number of test examples placed correctly in class
	vector<double> B(network.output.size(),0); //Number of test examples predicted to be in class but shouldn't
	vector<double> C(network.output.size(),0); //Number of test examples not predicted to be in class but should
	vector<double> D(network.output.size(),0); //Number of test examples predicted correctly to not be in class
	
	//Loop through each example
	for (int i = 0; i < examples.size(); ++i) {
		
		/*Propogate inputs forward*/
		
		//initialize input values
		for (int nodes = 0; nodes < network.input.size(); ++nodes) {
			network.input[nodes] = examples[i].x[nodes];
		}
		//Find value of hidden nodes from inputs
		for (int nodes = 0; nodes < network.hidden.size(); ++nodes) {
			double in = weightedsum(network.hid_in[nodes], network.input);
			network.hidden[nodes] = network.g(in);
		}
		//Find value of output nodes from hidden nodes
		for (int nodes = 0; nodes < network.output.size(); ++nodes) {
			double in = weightedsum(network.out_hid[nodes], network.hidden);
			network.output[nodes] = network.g(in);
		}
		
		/*Compare outputs*/
		for (int nodes = 0; nodes < network.output.size(); ++nodes) {
			int num = round(network.output[nodes]); //Round values to 1 or 0
			
			//Increment A, B, C, or D based on criteria
			if (num == 1 && examples[i].y[nodes] == 1)
				++A[nodes];
			else if (num == 1 && examples[i].y[nodes] == 0)
				++B[nodes];
			else if (num == 0 && examples[i].y[nodes] == 1)
				++C[nodes];
			else
				++D[nodes];
		}
	}
	
	//Output A,B,C,D values and metrics
	for (int nodes = 0; nodes < network.output.size(); ++nodes) {
		output.setf(ios::fixed,ios::floatfield);
		output.precision(0);
		output << A[nodes] << " " << B[nodes] << " " << C[nodes] << " " << D[nodes] << " ";
		
		vector<double> result = metrics(A[nodes],B[nodes],C[nodes],D[nodes]);
		
		output.setf(ios::fixed,ios::floatfield);
		output.precision(3);
		output << result[0] << " "; 
		output << result[1] << " ";
		output << result[2] << " "; 
		output << result[3] << endl;
	}
	
	//Find micro-average
	vector<double> result = metrics(sum(A),sum(B),sum(C),sum(D));
	
	//Output micro-average
	output.setf(ios::fixed,ios::floatfield);
	output.precision(3);
	output << result[0] << " "; 
	output << result[1] << " ";
	output << result[2] << " "; 
	output << result[3] << endl;
	
	//Find and output macro-average
	result = macro(A,B,C,D);
	output << result[0] << " "; 
	output << result[1] << " ";
	output << result[2] << " "; 
	output << result[3] << endl;
}

int main() {
	neuNet network; //neural network
	vector<in_out> examples; //testing data
	ofstream output; //output stream
	
	startNeural(network); //Initialize network
	startTesting(examples); //Initialize testing data
	startElse(output); //Open output file
	
	test(examples,network,output); //Use neural network on testing data
	
	output.close();
}
