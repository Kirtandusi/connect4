use crate::neuralnetwork::neuron::Neuron;

pub type Layer = Vec<Neuron>; //each layer is a vector of Neurons

#[derive(Clone)]
pub struct NeuralNetwork {
    pub(crate) layers: Vec<Layer>,
    pub(crate) learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>, learning_rate: f64) -> Self {
        Self {
            layers,
            learning_rate,
        }
    }
    /*
    calculates mean squared error between target and prediction. needed for debugging.
     */
    #[allow(dead_code)]
    fn mse_loss(target: &Vec<f64>, prediction: &Vec<f64>) -> f64 {
        target
            .iter()
            .zip(prediction.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            / target.len() as f64
    }
    /*
    forward pass. Calculates z = w * x + b. applies activation, stores output.
     */
    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let mut current_input = input.clone();

        for layer in &mut self.layers {
            let mut next_input = vec![];
            for neuron in layer {
                let z = neuron
                    .weights
                    .iter()
                    .zip(&current_input)
                    .map(|(w, i)| w * i)
                    .sum::<f64>()
                    + neuron.bias;
                neuron.z = z;
                neuron.output = (neuron.activation)(z);
                next_input.push(neuron.output);
            }
            current_input = next_input;
        }

        current_input
    }
    pub fn back(&mut self, input: &Vec<f64>, target: &Vec<f64>) {
        let mut deltas: Vec<Vec<f64>> = Vec::new();

        // 1. Compute deltas for output layer
        let output_layer = self.layers.last().unwrap();
        let mut output_deltas = Vec::with_capacity(output_layer.len());
        for (i, neuron) in output_layer.iter().enumerate() {
            let error = neuron.output - target[i];
            let delta = error * (neuron.activation_derivative)(neuron.z);
            output_deltas.push(delta);
        }
        deltas.push(output_deltas); // output layer deltas at index 0

        // 2. Compute deltas for hidden layers backwards
        // Layers before output layer: indices 0..(len-1)
        for l in (0..self.layers.len() - 1).rev() {
            let layer = &self.layers[l];
            let next_layer = &self.layers[l + 1];
            let next_deltas = deltas.last().unwrap(); // last computed deltas correspond to next_layer

            let mut layer_deltas = Vec::with_capacity(layer.len());
            for (i, neuron) in layer.iter().enumerate() {
                let mut sum = 0.0;
                for (j, next_neuron) in next_layer.iter().enumerate() {
                    sum += next_neuron.weights[i] * next_deltas[j];
                }
                let delta = sum * (neuron.activation_derivative)(neuron.z);
                layer_deltas.push(delta);
            }
            deltas.push(layer_deltas); // push new deltas at the end
        }

        // 3. Now deltas vector contains [output_layer, hidden_layer_1, hidden_layer_2, ..., input_layer] deltas
        // We want to update weights layer-by-layer in order: input layer to output layer
        // So reverse deltas to align with layers order:
        deltas.reverse();

        // 4. Update weights and biases
        let mut prev_output = input.clone();
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            for (neuron_index, neuron) in layer.iter_mut().enumerate() {
                let delta = deltas[layer_index][neuron_index];

                // Update weights
                for w in 0..neuron.weights.len() {
                    neuron.weights[w] -= self.learning_rate * delta * prev_output[w];
                }
                // Update bias
                neuron.bias -= self.learning_rate * delta;
            }
            // Prepare prev_output for next layer: outputs of current layer neurons
            prev_output = layer.iter().map(|n| n.output).collect();
        }
    }

}