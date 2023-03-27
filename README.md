Text Generator

This project uses deep learning to generate new text based on a given input text. In this example, we are using a pre-downloaded text of Shakespeare's works.

Installation

This project requires Python 3 and the following libraries:

numpy
tensorflow
keras

To install these libraries, you can use pip:

Copy code
pip install numpy tensorflow keras

Usage

To run the project, simply run the code in text_generator.py. This will train a model using the pre-downloaded text and save it as textgenerator.model.

To generate new text using the trained model, call the generate_text(length, temperature) function with the desired length of text and temperature. The function will randomly choose a seed from the pre-downloaded text and generate new text based on that seed and the trained model.

Example usage:

bash
Copy code
print('----- diversity: 0.2 -----')
print(generate_text(300, 0.2))  # generate 300 characters with a low temperature
print('----- diversity: 0.5 -----')
print(generate_text(300, 0.5))  # generate 300 characters with a medium temperature
print('----- diversity: 1.0 -----')
print(generate_text(300, 1.0))  # generate 300 characters with a high temperature
This will print out three examples of generated text with different levels of randomness (controlled by the temperature parameter).

Credits

This project is inspired by the TensorFlow tutorial on text generation (https://www.tensorflow.org/tutorials/text/text_generation). The pre-downloaded text of Shakespeare's works is provided by TensorFlow.
