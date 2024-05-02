#! /usr/bin/env python3

import onnx
import tensorrt as trt

#Create logger (associated with the builder and engine to capture errors, warnings, and other information during the build and inference phases):
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

if __name__ == '__main__':
    root_path = '/home/jetson/catkin_ws/src/mavros-px4-vehicle/models/'
    onnx_file = root_path + 'DQN_eval_learner.onnx'
    trt_file = root_path + 'DQN_eval_learner.trt'

    # Load the ONNX model:
    model = onnx.load(onnx_file)
   
    # Create TensorRT builder:
    builder = trt.Builder(TRT_LOGGER)

    # Create a TensorRT network object:
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Parse ONNX:
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    # Create a build configuration:
    config = builder.create_builder_config()
    # Set the maximum workspace size to 62 (max value):
    config.max_workspace_size = 1 << 62
        
    # Build and serialize the engine:
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trt_file, "wb") as f:
        f.write(engineString)
        
    # Print data types of layers in the engine:
    for layer in network:
        print(f"Layer Name: {layer.name}, Data Type: {layer.get_output(0).dtype}")
