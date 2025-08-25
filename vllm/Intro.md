

## vLLM Request Processing Workflow

The diagram above illustrates the complete request processing flow in vLLM. Here's how a request flows through the system:
<img width="686" height="1050" alt="image" src="https://github.com/user-attachments/assets/e0f58397-78d4-4525-a360-b4704c2a7b58" />


## 1. Entry Points

vLLM provides three main entry points:

- **`LLM` class** (`vllm/entrypoints/llm.py`): For offline batch inference
- **`AsyncLLMEngine`** (`vllm/engine/async_llm_engine.py`): For online serving with async capabilities  
- **OpenAI API Server** (`vllm/entrypoints/openai/api_server.py`): REST API compatible with OpenAI

## 2. Core Engine Layer

### LLMEngine (`vllm/engine/llm_engine.py`)
The heart of vLLM, responsible for:
- Managing the entire inference pipeline
- Coordinating between scheduler, executor, and output processor
- Implementing the main `step()` method that drives the inference loop

### AsyncLLMEngine (`vllm/engine/async_llm_engine.py`)
Wraps `LLMEngine` to provide:
- Asynchronous request handling
- Request streaming capabilities
- Non-blocking inference operations

## 3. Request Processing

### Request Ingestion
1. **`add_request()`**: Accepts incoming requests with prompts and sampling parameters
2. **Input Preprocessing** (`vllm/inputs/preprocess.py`): 
   - Tokenizes prompts using the configured tokenizer
   - Validates input parameters
   - Handles multimodal data (images, audio, etc.)
3. **Sequence Creation**: Creates `Sequence` and `SequenceGroup` objects that represent the request throughout its lifecycle

## 4. Scheduling Layer

### Scheduler (`vllm/core/scheduler.py`)
The scheduler is the brain of vLLM's efficiency:

- **Request Queuing**: Maintains queues for waiting, running, and swapped requests
- **Memory Management**: Works with `BlockSpaceManager` to allocate KV cache blocks
- **Batching Strategy**: Implements sophisticated batching algorithms including:
  - Continuous batching (processing new requests while others are generating)
  - Chunked prefill (breaking large prompts into chunks)
  - Preemption and swapping when memory is constrained

### Key Scheduling Components:
- **`SchedulingBudget`**: Tracks available tokens and sequence slots
- **`BlockSpaceManager`**: Manages KV cache block allocation/deallocation
- **`SchedulerOutputs`**: Contains the scheduling decision for the current step

## 5. Execution Layer

### Executor Hierarchy
```
ExecutorBase (Abstract)
├── UniProcExecutor (Single process)
├── RayDistributedExecutor (Ray-based distributed)
└── MPDistributedExecutor (Multiprocessing-based distributed)
```

### Workers and Model Runners
- **Workers** (`vllm/worker/worker.py`): Handle model execution on individual GPUs
- **Model Runners** (`vllm/worker/model_runner.py`): 
  - Prepare model inputs
  - Execute the actual model forward pass
  - Handle CUDA graphs for optimized execution

## 6. Model Inference

### Forward Pass Execution
1. **Input Preparation**: Convert scheduled sequences into model-compatible tensors
2. **Attention Computation**: Efficient attention mechanisms with KV caching
3. **Model Forward**: Execute the transformer layers
4. **Sampling**: Generate next tokens using configured sampling strategies

### Key Optimizations:
- **KV Cache Management**: Efficient memory usage through PagedAttention
- **CUDA Graphs**: For decode phases to reduce kernel launch overhead
- **Tensor Parallelism**: Distribute model across multiple GPUs
- **Pipeline Parallelism**: Process different model layers on different GPUs

## 7. Output Processing

### Post-Processing Pipeline
1. **`SamplerOutput`**: Raw model outputs with logits and token IDs
2. **Output Processor**: Handles finishing conditions, stop sequences
3. **Detokenization**: Convert token IDs back to text
4. **Response Formatting**: Create final `RequestOutput` or `PoolingRequestOutput`

## Key Class Hierarchy Relationships

### Core Classes:
- **`SequenceGroup`**: Groups related sequences (e.g., for beam search)
- **`Sequence`**: Individual sequence with prompt + generated tokens
- **`SamplingParams`**: Controls generation behavior (temperature, top_p, etc.)
- **`RequestOutput`**: Final output returned to user

### Configuration Classes:
- **`VllmConfig`**: Master configuration container
- **`ModelConfig`**: Model-specific settings
- **`SchedulerConfig`**: Scheduling policy and limits
- **`ParallelConfig`**: Distributed execution settings

## Request Lifecycle Example

1. **Request Arrival**: User calls `llm.generate(prompts, sampling_params)`
2. **Preprocessing**: Tokenize prompt, create `SequenceGroup`
3. **Scheduling**: Scheduler decides to process this request (or queue it)
4. **Execution**: 
   - Prefill phase: Process entire prompt in parallel
   - Decode phase: Generate tokens one by one
5. **Output**: Return `RequestOutput` with generated text

## Advanced Features

### Continuous Batching
Unlike traditional batching, vLLM can add new requests to existing batches, maximizing GPU utilization.

### Memory Efficiency
- **PagedAttention**: KV cache stored in non-contiguous blocks
- **Preemption**: Can pause low-priority requests to serve high-priority ones
- **Swapping**: Move KV cache between GPU and CPU memory

### Multi-Modal Support
The architecture extends to handle images, audio, and other modalities through specialized processors.

This architecture makes vLLM highly efficient for serving LLMs in production, with industry-leading throughput and latency characteristics through sophisticated memory management and request scheduling.
