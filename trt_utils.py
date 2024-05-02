import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda
import tensorrt as trt

def allocate_buffers(engine):
	inputs = []
	outputs = []
	bindings = []
	stream = cuda.Stream()
	for binding in engine:
		shape = engine.get_binding_shape(binding)
		size = trt.volume(shape)
		dtype = trt.nptype(engine.get_binding_dtype(binding))
		# Allocate host and device buffers
		host_mem = cuda.pagelocked_empty(size, dtype)
		device_mem = cuda.mem_alloc(host_mem.nbytes)
		# Append the device buffer to device bindings.
		bindings.append(int(device_mem))
		# Append to the appropriate list.
		if engine.binding_is_input(binding):
			inputs.append(EngineBuffer(host_mem, device_mem))
		else:
			outputs.append(EngineBuffer(host_mem, device_mem))
	return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
	'''This function copies the input data to the GPU, executes the model
	asynchronously, retrieves the output from the GPU, and returns the output data'''
	# Transfer input data to the GPU.
	[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
	# Run inference.
	context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
	# Transfer predictions back from the GPU.
	[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
	# Synchronize the stream
	stream.synchronize()
	# Return only the host outputs.
	return [out.host for out in outputs]
	
class EngineBuffer:
	def __init__(self, host_buffer, device_buffer):
		self.host = host_buffer
		self.device = device_buffer
