#! /usr/bin/env python3

import onnx
import tensorrt as trt

# Create logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

if __name__ == '__main__':
    root_path = '/home/jetson/catkin_ws/src/mavros-px4-vehicle/models/'
    onnx_file = root_path + 'D3QN_eval_half.onnx'
    trt_file = root_path + 'D3QN_eval_half.trt'

    # Load the ONNX model
    model = onnx.load(onnx_file)
   
    # Create TensorRT builder
    builder = trt.Builder(TRT_LOGGER)

    # Create a TensorRT network object
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Parse ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
        
    # Set the data type of each layer, input, and output in the network to FP16 where applicable
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.get_output(0).dtype == trt.DataType.FLOAT:
            layer.set_output_type(0, trt.DataType.HALF)
            layer.precision = trt.DataType.HALF

    for i in range(network.num_inputs):
        input_dtype = network.get_input(i).dtype
        if input_dtype == trt.DataType.FLOAT:
            network.get_input(i).dtype = trt.DataType.HALF

    for i in range(network.num_outputs):
        output_dtype = network.get_output(i).dtype
        if output_dtype == trt.DataType.FLOAT:
            network.get_output(i).dtype = trt.DataType.HALF

    # Create a build configuration
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 62
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Build and serialize the engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trt_file, "wb") as f:
        f.write(engineString)

