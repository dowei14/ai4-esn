/**
 * @author Sakya & Poramate 02.12.2013
 */

#include "esntest.h"
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */


//Add ENS network--(1)
#include "utils/esn-framework/networkmatrix.h"
//-----ESN network-----//
ESNetwork * ESN;
float * ESinput;
float * ESTrainOutput;

double mse;
double squared_error;






// ----------------------------------------------------------------------
// ------------ Initial constructor -------------------------------------
// ----------------------------------------------------------------------

TestESN::TestESN()
{

  srand (time(NULL));

  //--------------------------Add ESN network--(2)-----------------------------------//

  //For students
  ESN = new ESNetwork(num_input_ESN/*no. input*/,num_output_ESN /*no. output*/, num_hidden_ESN /*rc hidden neurons*/, false /*W_back, feedback from output to hiddens*/, false /*feeding input to output*/, leak /*leak = 0.0-1.0*/, false /*IP*/);
  ESN->outnonlinearity = 1; //0 = linear, 1 = sigmoid (logistic), 2  = tanh: transfer function of an output neuron
  ESN->nonlinearity = 2; //0 = linear, 1 = sigmoid (logistic), 2  = tanh: transfer function of all hidden neurons
  ESN->withRL = 2; //2 = stand ESN learning, 1 = RL with TD learning

  ESN->InputSparsity = input_sparsity; //if 0 = input connects to all hidden neurons, if 100 = input does not connect to hidden neurons
  ESN->autocorr = pow(10,4);//set as high as possible, default = 1
  ESN->InputWeightRange = 0.5; //Input scaling--> scaling of input to hidden neurons, default 0.15 means [-0.15, +0.15]
  ESN->LearnMode = learning_mode; //RLS = 1 (learning rate needs to be large, 0.99). LMS =2 (learning rate needs to be very small, e.g., 0.01)
  ESN->Loadweight = false; // true = loading learned weights
  ESN->NoiseRange = 0.0001;// amplitude of noise
  ESN->RCneuronNoise = true;//= constant fixed bias, true = changing noise bias every time

  ESN->generate_random_weights(50/*if 10 means 10% sparsity = 90% connectivity */, 0.9 /*Spectral radius < 1.0 to maintain echo state property, 1.2-1.5 = chaotic*/);

  washout_time = 500;


  //Create ESN input vector
  ESinput = new float[num_input_ESN];
  //Create ESN target output vector
  ESTrainOutput = new float[num_output_ESN];

  //Initial values of input and target output
  for(unsigned int i = 0; i < num_input_ESN; i++)
  {
    ESinput[i] = 0.0;
  }

  for(unsigned int i = 0; i< num_output_ESN; i++)
  {
    ESTrainOutput[i] = 0.0;
  }

  //inilialize mean squared error
  mse = 0.0;

  //--------------------------Add ESN network--(2)-----------------------------------//

  saveFile1.open("result.txt",ios::out);
  saveFile1.precision(5);



}

// ----------------------------------------------------------------------
// ------------ destructor ----------------------------------------------
// ----------------------------------------------------------------------

TestESN::~TestESN()
{
  delete []ESN;
  delete []ESinput;
  delete []ESTrainOutput;

}

// ----------------------------------------------------------------------
// --- Recurrent Neural Networks  ---------------------------------------
// ----------------------------------------------------------------------
double TestESN::RecurrentNetwork (std::vector<double> i0, std::vector<double> d, bool train)
{
  
	learn = true;
	static int iteration = 0;
	static float mse = 0.0;
	iteration++;

	if (train ==false)
	{
		learn = false;
	}
	else
	{
		mse = 0.0;
	}


	for (int i=0;i<num_input_ESN;i++){
		ESinput[i] = i0[i];
	}
	for (int i=0;i<num_output_ESN;i++){
		ESTrainOutput[i] = d[i];
	}

	ESN->setInput(ESinput, num_input_ESN/* no. input*/); // Call ESN

	//ESN Learning function
	ESN->takeStep(ESTrainOutput, learning_rate_ESN /*0.9 RLS*/, 1 /*no td = 1 else td_error*/, learn/* true= learn, false = not learning learn_critic*/, iteration/*0*/);
	unsigned int maxNum = 0;
	if (train == false){
		// only use max value
		
		double max = -2.0;
		for (unsigned int i=0;i<num_output_ESN;i++){
			output_ESN = ESN->outputs->val(i, 0);//Read out the output of ESN
			if (output_ESN > max) {
				maxNum =i;
				max = output_ESN;
			}
		}
		// Calculate online error at each time step
		double val = 0.0;
		squared_error = 0.0;
		for (unsigned int i=0;i<num_output_ESN;i++){
			if (i == maxNum) val = 1.0;
			else val = 0.0;
			output_ESN = ESN->outputs->val(i, 0);//Read out the output of ESN
			squared_error += (d[i]-val)*(d[i]-val);
			saveFile1<<val<<" ";
		}
		if (squared_error > 0) mse++;
		saveFile1<<"\n";
	}
	return maxNum;

}

bool TestESN::store()
{
	ESN->writeInnerweightsToFile(1);
    ESN->writeStartweightsToFile(1);
    ESN->writeEndweightsToFile(1);
    ESN->writeNoiseToFile(1);
}

bool TestESN::load()
{
    ESN->readStartweightsFromFile(1);
    ESN->readEndweightsFromFile(1);
    ESN->readInnerweightsFromFile(1);
    ESN->readNoiseFromFile(1);
}
