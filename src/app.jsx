import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { motion, AnimatePresence } from 'framer-motion';

const App = () => {
  const [section, setSection] = useState('intro');
  const [mnistData, setMnistData] = useState(null);
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [currentDigit, setCurrentDigit] = useState(0);

  // Load MNIST dataset
  useEffect(() => {
    async function loadMnist() {
      const data = await tf.data.csv('https://storage.googleapis.com/tfjs-tutorials/mnist.csv');
      const images = [];
      const labels = [];
      await data.forEachAsync(row => {
        const img = Object.values(row).slice(1).map(v => v / 255);
        images.push(img);
        labels.push(row.label);
      });
      setMnistData({ images: tf.tensor2d(images), labels: tf.tensor1d(labels, 'int32') });
    }
    loadMnist();
  }, []);

  // Build and train model
  useEffect(() => {
    if (mnistData) {
      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 512, activation: 'relu', inputShape: [784] }));
      model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
      model.compile({
        optimizer: 'rmsprop',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy'],
      });
      model.fit(mnistData.images, mnistData.labels, {
        epochs: 5,
        batchSize: 128,
        callbacks: {
          onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}`),
        },
      }).then(() => setModel(model));
    }
  }, [mnistData]);

  // Make predictions
  const predict = async (index) => {
    if (model && mnistData) {
      const digit = mnistData.images.slice([index, 0], [1, 784]);
      const pred = await model.predict(digit).data();
      setPredictions(Array.from(pred));
      setCurrentDigit(index);
    }
  };

  const sections = {
    intro: (
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        className="min-h-screen flex flex-col items-center justify-center text-center p-8"
      >
        <h1 className="text-6xl font-playfair text-grammy-gold mb-4">Neural Networks Unveiled</h1>
        <p className="text-2xl mb-8">A glamorous journey through deep learning with MNIST</p>
        <motion.button
          whileHover={{ scale: 1.1, boxShadow: '0 0 20px rgba(212, 175, 55, 0.8)' }}
          className="px-6 py-3 bg-grammy-gold text-grammy-black rounded-full text-lg"
          onClick={() => setSection('mnist')}
        >
          Start the Show
        </motion.button>
      </motion.div>
    ),
    mnist: (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="min-h-screen p-8"
      >
        <h2 className="text-4xl font-playfair text-grammy-gold mb-4">Loading the MNIST Dataset</h2>
        <p className="text-lg mb-4">Explore handwritten digits (28x28 pixels) with 60,000 training and 10,000 test images.</p>
        {mnistData && (
          <div className="flex gap-4">
            <canvas
              ref={(canvas) => {
                if (canvas && mnistData) {
                  const ctx = canvas.getContext('2d');
                  const img = mnistData.images.slice([currentDigit, 0], [1, 784]).reshape([28, 28]).arraySync();
                  const imageData = ctx.createImageData(28, 28);
                  for (let i = 0; i < 28; i++) {
                    for (let j = 0; j < 28; j++) {
                      const idx = (i * 28 + j) * 4;
                      const val = img[i][j] * 255;
                      imageData.data[idx] = val;
                      imageData.data[idx + 1] = val;
                      imageData.data[idx + 2] = val;
                      imageData.data[idx + 3] = 255;
                    }
                  }
                  ctx.putImageData(imageData, 0, 0);
                }
              }}
              width="28"
              height="28"
              className="border-2 border-grammy-gold"
              style={{ imageRendering: 'pixelated', width: '112px', height: '112px' }}
            />
            <div>
              <p>Label: {mnistData.labels.arraySync()[currentDigit]}</p>
              <input
                type="range"
                min="0"
                max="9999"
                value={currentDigit}
                onChange={(e) => setCurrentDigit(Number(e.target.value))}
                className="w-64"
              />
            </div>
          </div>
        )}
        <motion.button
          whileHover={{ scale: 1.1 }}
          className="mt-4 px-4 py-2 bg-grammy-gold text-grammy-black rounded"
          onClick={() => setSection('architecture')}
        >
          Next: Network Architecture
        </motion.button>
      </motion.div>
    ),
    architecture: (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="min-h-screen p-8"
      >
        <h2 className="text-4xl font-playfair text-grammy-gold mb-4">Neural Network Architecture</h2>
        <p className="text-lg mb-4">A sequential model with 512 ReLU neurons and 10 softmax outputs.</p>
        <motion.div
          className="flex justify-center"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
        >
          <svg width="600" height="200" className="border-2 border-grammy-gold">
            {/* Input Layer */}
            {Array(10).fill().map((_, i) => (
              <circle key={`input-${i}`} cx="50" cy={20 + i * 15} r="5" fill="white" />
            ))}
            <text x="30" y="10" fill="white">Input (784)</text>
            {/* Hidden Layer */}
            {Array(10).fill().map((_, i) => (
              <circle key={`hidden-${i}`} cx="300" cy={20 + i * 15} r="5" fill="white" />
            ))}
            <text x="280" y="10" fill="white">Hidden (512, ReLU)</text>
            {/* Output Layer */}
            {Array(10).fill().map((_, i) => (
              <circle key={`output-${i}`} cx="550" cy={20 + i * 15} r="5" fill="white" />
            ))}
            <text x="530" y="10" fill="white">Output (10, Softmax)</text>
            {/* Connections */}
            {Array(10).fill().map((_, i) =>
              Array(10).fill().map((_, j) => (
                <line
                  key={`conn-${i}-${j}`}
                  x1="50" y1={20 + i * 15}
                  x2="300" y2={20 + j * 15}
                  stroke="white" strokeWidth="0.5" opacity="0.3"
                />
              ))
            )}
            {Array(10).fill().map((_, i) =>
              Array(10).fill().map((_, j) => (
                <line
                  key={`conn2-${i}-${j}`}
                  x1="300" y1={20 + i * 15}
                  x2="550" y2={20 + j * 15}
                  stroke="white" strokeWidth="0.5" opacity="0.3"
                />
              ))
            )}
          </svg>
        </motion.div>
        <motion.button
          whileHover={{ scale: 1.1 }}
          className="mt-4 px-4 py-2 bg-grammy-gold text-grammy-black rounded"
          onClick={() => setSection('training')}
        >
          Next: Training the Model
        </motion.button>
      </motion.div>
    ),
    training: (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="min-h-screen p-8"
      >
        <h2 className="text-4xl font-playfair text-grammy-gold mb-4">Training the Model</h2>
        <p className="text-lg mb-4">Train with RMSprop, sparse categorical crossentropy, and 5 epochs.</p>
        <motion.div
          className="w-64 h-4 bg-gray-700 rounded"
          initial={{ width: 0 }}
          animate={{ width: '100%' }}
          transition={{ duration: 5 }}
        >
          <div className="h-full bg-grammy-gold rounded" />
        </motion.div>
        <p className="mt-2">Training in progress...</p>
        <motion.button
          whileHover={{ scale: 1.1 }}
          className="mt-4 px-4 py-2 bg-grammy-gold text-grammy-black rounded"
          onClick={() => setSection('tensors')}
        >
          Next: Tensor Operations
        </motion.button>
      </motion.div>
    ),
    tensors: (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="min-h-screen p-8"
      >
        <h2 className="text-4xl font-playfair text-grammy-gold mb-4">Tensor Operations</h2>
        <p className="text-lg mb-4">Explore scalars, vectors, matrices, and higher-rank tensors.</p>
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className="flex gap-4"
        >
          <div>
            <p>Scalar (Rank-0)</p>
            <div className="w-12 h-12 bg-white text-grammy-black flex items-center justify-center">12</div>
          </div>
          <div>
            <p>Vector (Rank-1)</p>
            <div className="flex gap-1">
              {[12, 3, 6, 14, 7].map((v, i) => (
                <div key={i} className="w-12 h-12 bg-white text-grammy-black flex items-center justify-center">{v}</div>
              ))}
            </div>
          </div>
          <div>
            <p>Matrix (Rank-2)</p>
            <div className="grid grid-cols-5 gap-1">
              {[[5, 78, 2, 34, 0], [6, 79, 3, 35, 1], [7, 80, 4, 36, 2]].flat().map((v, i) => (
                <div key={i} className="w-12 h-12 bg-white text-grammy-black flex items-center justify-center">{v}</div>
              ))}
            </div>
          </div>
        </motion.div>
        <motion.button
          whileHover={{ scale: 1.1 }}
          className="mt-4 px-4 py-2 bg-grammy-gold text-grammy-black rounded"
          onClick={() => setSection('gradients')}
        >
          Next: Gradient-Based Optimization
        </motion.button>
      </motion.div>
    ),
    gradients: (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="min-h-screen p-8"
      >
        <h2 className="text-4xl font-playfair text-grammy-gold mb-4">Gradient-Based Optimization</h2>
        <p className="text-lg mb-4">Visualize gradients with TensorFlow's Gradient Tape.</p>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <svg width="300" height="200" className="border-2 border-grammy-gold">
            <circle cx="50" cy="150" r="10" fill="white" />
            <text x="30" y="140" fill="white">x</text>
            <line x1="50" y1="150" x2="150" y2="100" stroke="white" />
            <circle cx="150" cy="100" r="10" fill="white" />
            <text x="130" y="90" fill="white">2x</text>
            <line x1="150" y1="100" x2="250" y2="50" stroke="white" />
            <circle cx="250" cy="50" r="10" fill="white" />
            <text x="230" y="40" fill="white">y = 2x + 3</text>
          </svg>
        </motion.div>
        <motion.button
          whileHover={{ scale: 1.1 }}
          className="mt-4 px-4 py-2 bg-grammy-gold text-grammy-black rounded"
          onClick={() => setSection('predictions')}
        >
          Next: Predictions
        </motion.button>
      </motion.div>
    ),
    predictions: (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="min-h-screen p-8"
      >
        <h2 className="text-4xl font-playfair text-grammy-gold mb-4">Predictions and Evaluation</h2>
        <p className="text-lg mb-4">Test the model on new digits.</p>
        <button
          className="px-4 py-2 bg-grammy-gold text-grammy-black rounded"
          onClick={() => predict(currentDigit)}
        >
          Predict Digit {currentDigit}
        </button>
        {predictions.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-4"
          >
            <p>Predictions:</p>
            <div className="flex gap-2">
              {predictions.map((p, i) => (
                <div key={i} className="w-12 h-12 bg-white text-grammy-black flex items-center justify-center">
                  {i}: {p.toFixed(2)}
                </div>
              ))}
            </div>
            <p>Predicted Label: {predictions.indexOf(Math.max(...predictions))}</p>
          </motion.div>
        )}
        <motion.button
          whileHover={{ scale: 1.1 }}
          className="mt-4 px-4 py-2 bg-grammy-gold text-grammy-black rounded"
          onClick={() => setSection('summary')}
        >
          Next: Summary
        </motion.button>
      </motion.div>
    ),
    summary: (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="min-h-screen flex flex-col items-center justify-center text-center p-8"
      >
        <h2 className="text-4xl font-playfair text-grammy-gold mb-4">Summary</h2>
        <p className="text-lg mb-4">You've mastered the building blocks of neural networks!</p>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-left"
        >
          <ul className="list-disc pl-6">
            <li>Loaded and preprocessed the MNIST dataset.</li>
            <li>Built a neural network with dense layers.</li>
            <li>Trained the model using gradient descent.</li>
            <li>Explored tensor operations and shapes.</li>
            <li>Understood gradients and backpropagation.</li>
            <li>Made predictions with confidence.</li>
          </ul>
        </motion.div>
        <motion.button
          whileHover={{ scale: 1.1 }}
          className="mt-4 px-4 py-2 bg-grammy-gold text-grammy-black rounded"
          onClick={() => setSection('intro')}
        >
          Replay the Show
        </motion.button>
      </motion.div>
    ),
  };

  return (
    <AnimatePresence>
      {sections[section]}
    </AnimatePresence>
  );
};

export default App;
