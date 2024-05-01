#! /usr/bin/env python3

from dueling_deep_q_network import DuelingDeepQNetwork
from deep_q_network_learner import DeepQNetworkLearner
import torch
import onnx

if __name__ == '__main__':
    lr = 0.00025
    n_actions = 3
    image_input_dims = (3, 80, 106)  # channels, height, width
    root_path = '/home/pm/catkin_ws/src/mavros-px4-vehicle/models/'
    load_file = root_path + 'D3QN_eval.pth' #'DQN_eval_learner.pth'
    onnx_file = root_path + 'D3QN_eval.onnx' #'DQN_eval_learner.onnx'
    
    fc1 = 128#1024#128
    fc2 = 32#512#32
    
    # Create/load pruned model:
    '''model = DuelingDeepQNetwork(lr=lr,
                                n_actions=n_actions,
                                input_dims=image_input_dims,
                                fc1_out=fc1, fc2_out=fc2)'''
                                
    model = DeepQNetworkLearner(lr=lr,
                                n_actions=n_actions,
                                input_dims=image_input_dims,
                                fc1_out=fc1, fc2_out=fc2)
    model.load_checkpoint(load_file)  # Load the pruned model

    # Move model to CPU (just for now can move to GPU later)
    model = model.to('cpu')
    
    # Put model in evaluation mode
    model = model.eval()

    # Create example data and ensure it's on the CPU
    input_data = torch.ones((1, *image_input_dims)).to('cpu')
    
    # Export the trained model to ONNX
    print("Export trained model to ONNX")
    torch.onnx.export(model, input_data, onnx_file)
       
    ##################Print stuff######################
    # Load the ONNX model
    print("Load model and start printing stuff")
    onnx_model = onnx.load(onnx_file)

    # Print layer information
    print("Layers in ONNX model:")
    for layer in onnx_model.graph.node:
        print(f"Layer Name: {layer.name}, Layer Type: {layer.op_type}")
        for input in layer.input:
           print(f"\tInput: {input}")
        for output in layer.output:
            print(f"\tOutput: {output}")

    # Print weight data types
    print("\nWeights in ONNX model:")
    for weight in onnx_model.graph.initializer:
        print(f"Weight Name: {weight.name}, Data Type: {onnx.TensorProto.DataType.Name(weight.data_type)}")
    #####################################################
