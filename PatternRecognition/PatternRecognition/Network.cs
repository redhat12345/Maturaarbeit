using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace PatternRecognition
{
        public class NeuralNetwork
        {
            public double LearnRate { get; set; }
            public double Momentum { get; set; }
            public List<Neuron> InputLayer { get; set; }
            public List<Neuron> HiddenLayer { get; set; }
            public List<Neuron> OutputLayer { get; set; }
            static Random random = new Random();

            public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, TextBox txtLearn)
            {
                LearnRate = float.Parse(txtLearn.Text);
                Momentum = 0.1;
                InputLayer = new List<Neuron>();
                HiddenLayer = new List<Neuron>();
                OutputLayer = new List<Neuron>();

                for (int i = 0; i < inputSize; i++)
                    InputLayer.Add(new Neuron());

                for (int i = 0; i < hiddenSize; i++)
                    HiddenLayer.Add(new Neuron(InputLayer));

                for (int i = 0; i < outputSize; i++)
                    OutputLayer.Add(new Neuron(HiddenLayer));
            }

            //Sends the inputs once through the network
            public void Train(params double[] inputs)
            {
                int i = 0;
                InputLayer.ForEach(a => a.Value = inputs[i++]); //Assign input data to input-neurons
                HiddenLayer.ForEach(a => a.CalculateValue());   //Hidden Calc
                OutputLayer.ForEach(a => a.CalculateValue());   //Outuput Calc 
            }

            //Sends the inputs once through the network and returns the output
            public double[] Compute(params double[] inputs)
            {
                Train(inputs);
                return OutputLayer.Select(a => a.Value).ToArray();
            }

            public double CalculateError(params double[] targets)
            {
                int i = 0;
                return OutputLayer.Sum(a => Math.Abs(a.CalculateError(targets[i++])));
            }

            //Sends the outputs once backwards through the network to adjust the weights
            public void BackPropagate(params double[] targets)
            {
                int i = 0;
                OutputLayer.ForEach(a => a.CalculateGradient(targets[i++]));    //Gradient of output layer
                HiddenLayer.ForEach(a => a.CalculateGradient());                //Gradient of hidden layer
                HiddenLayer.ForEach(a => a.UpdateWeights(LearnRate, Momentum)); //Adjust weights in hidden layer
                OutputLayer.ForEach(a => a.UpdateWeights(LearnRate, Momentum)); //Adjust weights in hidden layer
            }

            //Returns a random number whereby x € [-1,1]
            public static double NextRandom()
            {
                return 2 * random.NextDouble() - 1;
            }

            public static double SigmoidFunction(double x)
            {
                //The first part is to round the values at a reasonable point
                if (x < -45.0)
                    return 0.0;
                else if (x > 45.0)
                    return 1.0;

                //actual function
                return 1.0 / (1.0 + Math.Exp(-x));
            }

            public static double SigmoidDerivative(double f)
            {
                return f * (1 - f);
            }
        }

        public class Neuron
        {
            public List<Synapse> IncomingSynapses { get; set; }
            public List<Synapse> OutgoingSynapses { get; set; }
            public double Bias { get; set; }
            public double BiasDelta { get; set; }
            public double Gradient { get; set; }
            public double Value { get; set; }

            //Constructor for input neurons
            public Neuron()
            {
                IncomingSynapses = new List<Synapse>();
                OutgoingSynapses = new List<Synapse>();
                Bias = NeuralNetwork.NextRandom();
            }

            //Constructor for other neurons        
            //First the other constructor is executed thanks to :this() "self-inheritance"
            public Neuron(List<Neuron> previousNeurons) : this()
            {
                foreach (var previousNeuron in previousNeurons)
                {
                    var synapse = new Synapse(previousNeuron, this);
                    previousNeuron.OutgoingSynapses.Add(synapse);
                    IncomingSynapses.Add(synapse);
                }
            }

            public virtual double CalculateValue()
            {
                return Value = NeuralNetwork.SigmoidFunction(IncomingSynapses.Sum(a => a.Weight * a.FromNeuron.Value) + Bias);
            }

            public virtual double CalculateDerivative()
            {
                return NeuralNetwork.SigmoidDerivative(Value);
            }

            public double CalculateError(double target)
            {
                return target - Value;
            }

            //Gradient in output neurons
            public double CalculateGradient(double target)
            {
                return Gradient = CalculateError(target) * CalculateDerivative();
            }

            //Gradient in hidden neurons
            public double CalculateGradient()
            {
                return Gradient = OutgoingSynapses.Sum(a => a.ToNeuron.Gradient * a.Weight) * CalculateDerivative();
            }

            public void UpdateWeights(double learnRate, double momentum)
            {
                var prevDelta = BiasDelta;
                BiasDelta = learnRate * Gradient; // * 1
                Bias += BiasDelta + momentum * prevDelta;

                foreach (var s in IncomingSynapses)
                {
                    prevDelta = s.WeightDelta;
                    s.WeightDelta = learnRate * Gradient * s.FromNeuron.Value;
                    //s.FromNeuron.Value technically also belongs to the gradient
                    //For output-weights: (Formula) dW = Alpha * Error * SigmoidDerivative * PreviousOutput
                    //                       (Code) dW = learnRate * (target - Value) * Value * (1 - Value) * s.FromNeuron.Value
                    s.Weight += s.WeightDelta + momentum * prevDelta;
                }
            }
        }
       
        public class Synapse
        {
            public Neuron FromNeuron { get; set; }
            public Neuron ToNeuron { get; set; }
            public double Weight { get; set; }
            public double WeightDelta { get; set; }

            public Synapse(Neuron fromNeuron, Neuron toNeuron)
            {
                FromNeuron = fromNeuron;
                ToNeuron = toNeuron;
                Weight = NeuralNetwork.NextRandom();
            }
        }
    
}
