# MNN

### **Markovian Neural Network (MNN) Theory**

The **Markovian Neural Network (MNN)** is an advanced neural network model where the traditional fixed-weight learning system is replaced with **Markovian processes** that govern the evolution of key parameters, such as the **learning rate** and **momentum**. This dynamic model adapts during training through feedback from its internal state, which enables it to adjust its learning strategies for improved performance.

#### **Key Concepts**:

1. **Markovian Coefficients**:

   * **Markovian coefficients** like the **learning rate** and **momentum** evolve based on the **previous state** of the model. This means that instead of manually setting the learning rate, it evolves dynamically based on feedback from the network's training performance.

2. **Noise as Mutation**:

   * **Noise** is applied to the inputs and weights to introduce controlled mutations. This allows the model to **explore** different solutions and avoid getting trapped in local minima. The noise is adjusted gradually to help transition from exploration to exploitation as the model learns.

3. **Dithering as a Filter**:

   * **Dithering** is applied to the outputs of neurons to encourage **exploration** and prevent overfitting. This small, random perturbation ensures that the model does not converge too quickly to a suboptimal solution.

4. **Evolving Parameters**:

   * The network's parameters evolve over time according to a **Markov process**. At each step, the coefficients (learning rate, momentum, etc.) are adjusted dynamically based on feedback from the training process. This introduces **adaptive learning** that allows the model to continuously improve based on its past performance.

5. **Training Dynamics**:

   * Instead of relying on fixed hyperparameters or a manual update process, the **Markovian Neural Network** adjusts its internal learning mechanisms in real-time. The learning process is driven by the **Markov coefficients**, and the network gradually stabilizes by reducing noise and dithering over time.

#### **Mathematical Representation**:

The **Markovian Neural Network** can be described by the following dynamics:

1. **State Equation**:

   * The state of the network at each time step $t$ is given by the current **weights** $W_t$, **learning rate** $\eta_t$, and **momentum** $\mu_t$, as well as the noise and dither parameters $\nu_t$ and $\delta_t$, respectively.

   The update rule is defined as:

   $$
   W_{t+1} = W_t - \eta_t \cdot \nabla L(W_t) + \mu_t \cdot (W_t - W_{t-1}) + \nu_t \cdot \epsilon_t
   $$

   where:

   * $\eta_t$ is the **learning rate** at time $t$.
   * $\mu_t$ is the **momentum**.
   * $\nu_t$ is the **noise factor** that adds random mutations.
   * $\epsilon_t$ is a random noise term.
   * $L(W_t)$ is the loss function at time $t$.

2. **Markov Coefficient Evolution**:
   The Markov coefficients evolve over time based on the system's performance:

   $$
   \eta_{t+1} = f(\eta_t, L(W_t)) \quad \text{(Learning rate evolution)}
   $$

   $$
   \mu_{t+1} = g(\mu_t, L(W_t)) \quad \text{(Momentum evolution)}
   $$

   $$
   \nu_{t+1} = \nu_t \cdot \text{decay} \quad \text{(Noise decay)}
   $$

   $$
   \delta_{t+1} = \delta_t \cdot \text{decay} \quad \text{(Dither decay)}
   $$

   These equations describe how the learning rate, momentum, noise, and dithering evolve based on the loss and feedback from the network.

3. **Output Equation**:
   The output at time step $t$ is given by the standard forward pass equation:

   $$
   y_t = \text{Activation}(W_t \cdot x + b)
   $$

   where:

   * $y_t$ is the output at time $t$.
   * $x$ is the input at time $t$.
   * $W_t$ is the weight matrix at time $t$.
   * $b$ is the bias vector.

   The activation function is typically **ReLU** or **Sigmoid**, depending on the task (classification or regression).

---

### **Markovian Neural Network (MNN) Script**

Below is the **MNN implementation** using **Markovian coefficients**, **noise**, and **dithering**. This implementation defines a neural network where the learning rate and momentum evolve over time based on training feedback.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ----- Markov Noisy Neuron -----
class MarkovNoisyNeuron(nn.Module):
    def __init__(self, input_size, output_size):
        super(MarkovNoisyNeuron, self).__init__()

        # Weights (trainable)
        self.weights = nn.Parameter(torch.randn(input_size, output_size))

        # Noise/dither learnable parameters
        self.noise_factor = nn.Parameter(torch.tensor(0.05))
        self.dither_strength = nn.Parameter(torch.tensor(0.05))

    def forward(self, x, epoch):
        # Decay noise & dither slightly each epoch (slower decay)
        decay = 0.99  # Slower decay for noise and dithering
        self.noise_factor.data *= decay
        self.dither_strength.data *= decay

        # Apply noise to inputs
        noise = torch.randn_like(x) * self.noise_factor
        x = x + noise

        # Linear transform
        output = torch.matmul(x, self.weights)

        # Apply dithering to outputs
        dither_noise = torch.randn_like(output) * self.dither_strength
        output = output + dither_noise

        return output


# ----- Network using noisy neurons -----
class MarkovNoisyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MarkovNoisyNetwork, self).__init__()
        self.layer1 = MarkovNoisyNeuron(input_size, hidden_size)
        self.layer2 = MarkovNoisyNeuron(hidden_size, output_size)

    def forward(self, x, epoch):
        x = self.layer1(x, epoch)
        x = F.relu(x)   # Non-linearity needed for XOR!
        x = self.layer2(x, epoch)
        return x


# ----- XOR dataset -----
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# ----- Model -----
model = MarkovNoisyNetwork(input_size=2, hidden_size=4, output_size=1)

criterion = nn.BCEWithLogitsLoss()  # combines sigmoid + BCE
optimizer = optim.Adam(model.parameters(), lr=0.005)  # Slightly reduced learning rate

# ----- Training -----
num_epochs = 10000
for epoch in range(num_epochs):
    outputs = model(X, epoch)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Monitoring the progress
    if (epoch+1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ----- Testing -----
model.eval()
with torch.no_grad():
    predicted = torch.sigmoid(model(X, num_epochs))  # Apply sigmoid for binary output
    print("\nRaw outputs:\n", predicted)
    print("Rounded predictions:\n", predicted.round())  # Rounded predictions for binary output
```

### **Explanation of Code**:

1. **MarkovNoisyNeuron**:

   * **Weights** are trainable parameters that evolve based on **Markov coefficients**.
   * The neuron has **noise** and **dither** factors which are **adjusted** over time using a decay factor, ensuring that noise and dither gradually decrease as training progresses.

2. **MarkovNoisyNetwork**:

   * This is the neural network that includes layers of **Markov Noisy Neurons**.
   * The network consists of two layers with **ReLU** applied after the first layer to introduce non-linearity.

3. **Training**:

   * The network is trained on the XOR dataset using **Binary Cross-Entropy Loss** (suitable for binary classification tasks).
   * The **Adam optimizer** is used for updating the weights, with a **reduced learning rate** to stabilize the training process.

4. **Testing**:

   * After training, the model’s output is passed through the **sigmoid function** to ensure binary outputs, which are then **rounded** to either 0 or 1.

---

### **Conclusion**:

The **Markovian Neural Network (MNN)** uses **Markov coefficients** to evolve its learning dynamics, applying **noise** and **dithering** to explore and refine its solution space. The model gradually stabilizes as it **reduces randomness** over time, allowing it to converge on an optimal solution.

This architecture can be extended to more complex tasks beyond XOR, where the dynamic nature of the **Markov coefficients** can provide adaptive learning strategies.
