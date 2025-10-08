<h1>Simple Neural Network</h1>

<p>
  A fully functional neural network built <strong>from scratch using NumPy</strong>, designed for educational purposes.
  It solves both the <strong>XOR problem</strong> and a <strong>spiral classification task</strong>,
  with detailed visualizations of predictions and decision boundaries.
</p>

<hr />

<h2>Learning objectives</h2>
<ul>
  <li>Understand the <strong>mathematical foundations</strong> of neural networks without high-level ML libraries.</li>
  <li>Implement <strong>forward propagation</strong>, <strong>backpropagation</strong>, and <strong>gradient descent</strong> manually.</li>
  <li>Visualize how a neural network learns to classify complex, non-linear data like XOR and spirals.</li>
  <li>Design flexible neural architectures with custom layers and activation logic.</li>
</ul>

<h2>Brief description of the model</h2>
<ul>
  <li><strong>Custom architecture</strong>: Configurable layers (e.g. [2, 4, 1] or [2, 10, 8, 1])</li>
  <li><strong>Sigmoid activation</strong> for all layers</li>
  <li><strong>Mean Squared Error (MSE)</strong> loss function</li>
  <li><strong>Xavier initialization</strong> for weights</li>
  <li>Two datasets:
    <ul>
      <li>XOR (non-linear logic gate)</li>
      <li>2D Spiral (complex shape classification)</li>
    </ul>
  </li>
</ul>

<h2>🛠️ Technical stack</h2>
<ul>
  <li><strong>Python 3.10+</strong></li>
  <li><strong>NumPy</strong> – Vectorized operations</li>
  <li><strong>Matplotlib</strong> – Plotting & Visualization</li>
  <li>No external ML frameworks used</li>
</ul>

<h2>🔭 Future developments and improvements</h2>
<ul>
  <li>Add support for other activation functions (ReLU, tanh)</li>
  <li>Enable multiclass classification support</li>
  <li>Implement mini-batch training</li>
  <li>Explore new datasets</li>
  <li>Add real-time dashboards with Streamlit or Gradio</li>
</ul>

<h2>How to run</h2>
<pre><code>pip install numpy matplotlib
python simple_neural_network.py
</code></pre>

<p>This script will:</p>
<ul>
  <li>Train a neural network on the <strong>XOR dataset</strong></li>
  <li>Train on a <strong>2D spiral dataset</strong></li>
  <li>Plot decision boundaries and loss curves</li>
</ul>

<h2>📸 Sample output</h2>

![Figure_3](https://github.com/user-attachments/assets/8a4886db-7bd9-4b54-aaef-441e09ee0cfb)

![Figure_2](https://github.com/user-attachments/assets/e0c85aec-00cb-4173-8933-45db4cbd1b70)

![Figure_1](https://github.com/user-attachments/assets/af9b75bb-b0d2-4bab-9d4b-9aedfe147e4a)



<hr />
<blockquote>
  <p>This project is a hands-on demonstration of how neural networks work at the lowest level. It’s ideal for students, curious developers, and anyone seeking to learn deep learning fundamentals without the abstraction of modern ML libraries.</p>
</blockquote>
