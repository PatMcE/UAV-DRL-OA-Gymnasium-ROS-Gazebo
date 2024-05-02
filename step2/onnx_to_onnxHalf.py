#! /usr/bin/env python3

import onnx
import numpy as np
from onnx import numpy_helper

# Function to convert a tensor from float32 to float16
def convert_tensor(tensor):
    if tensor.data_type == onnx.TensorProto.FLOAT:
        float_array = numpy_helper.to_array(tensor)
        float16_array = float_array.astype(np.float16)
        tensor.CopyFrom(numpy_helper.from_array(float16_array, tensor.name))

def print_tensor_data_types(model):
    for initializer in model.graph.initializer:
        tensor_data_type = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        print(f"Name: {initializer.name}, Type: {tensor_data_type}")

if __name__ == '__main__':
    root_path = '/home/pm/catkin_ws/src/mavros-px4-vehicle/models/'
    onnx_filepath = root_path + 'D3QN_eval.onnx' #'DQN_eval_learner.onnx'
    onnx_fp16_filepath = root_path + 'D3QN_eval_half.onnx' #'DQN_eval_learner_half.onnx'

    # Load the ONNX model
    print('Load model')
    model = onnx.load(onnx_filepath)

    # Print the data types of the weights and biases
    print('Before conversion model:')
    print_tensor_data_types(model)

    print('Converting ...')
    # Convert all initializers (weights) to float16
    for initializer in model.graph.initializer:
        convert_tensor(initializer)

    # Convert all inputs to float16
    for input in model.graph.input:
        if input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            input.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

    # Convert all outputs to float16
    for output in model.graph.output:
        if output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

    # Save the converted FP16 model
    print('Save converted model')
    onnx.save(model, onnx_fp16_filepath)

    # Load the new ONNX model
    print('Load converted model')
    model = onnx.load(onnx_fp16_filepath)

    # Print the data types of the weights and biases
    print('After conversion model:')
    print_tensor_data_types(model)

