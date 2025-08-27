# install vLLM

```bash
python /home/jupyter/.local/lib/python3.10/site-packages/uv/__main__.py pip install vllm --torch-backend=auto
pip install ipython
pip install flashinfer-python

# start the API server
/libraries/vllmenv/bin/vllm serve /home/jupyter/models/Meta-Llama-3.1-8B-Instruct --dtype float16 --port 8010 --disable-log-requests --tensor-parallel-size 1

# sample log
Testing Get Stats Script....
Started Running GPU stats
INFO 08-27 21:02:11 [__init__.py:241] Automatically detected platform cuda.
WARNING 08-27 21:02:14 [__init__.py:1726] argument '--disable-log-requests' is deprecated and replaced with '--enable-log-requests'. This will be removed in v0.12.0.
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:02:14 [api_server.py:1805] vLLM API server version 0.10.1.1
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:02:14 [utils.py:326] non-default args: {'model_tag': '/home/jupyter/models/Meta-Llama-3.1-8B-Instruct', 'port': 8010, 'model': '/home/jupyter/models/Meta-Llama-3.1-8B-Instruct', 'dtype': 'float16'}
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:02:22 [__init__.py:711] Resolved architecture: *LlamaForCausalLM*
[1;36m(APIServer pid=15)[0;0m WARNING 08-27 21:02:22 [__init__.py:2819] Casting torch.bfloat16 to torch.float16.
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:02:22 [__init__.py:1750] Using max model len 131072
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:02:22 [scheduler.py:222] Chunked prefill is enabled with max_num_batched_tokens=2048.
INFO 08-27 21:02:28 [__init__.py:241] Automatically detected platform cuda.
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:30 [core.py:636] Waiting for init message from front-end.
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:30 [core.py:74] Initializing a V1 LLM engine (v0.10.1.1) with config: model='/home/jupyter/models/Meta-Llama-3.1-8B-Instruct', speculative_config=None, tokenizer='/home/jupyter/models/Meta-Llama-3.1-8B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=131072, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/home/jupyter/models/Meta-Llama-3.1-8B-Instruct, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":1,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":null}
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:33 [parallel_state.py:1134] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
[1;36m(EngineCore_0 pid=82)[0;0m WARNING 08-27 21:02:33 [topk_topp_sampler.py:61] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:33 [gpu_model_runner.py:1953] Starting to load model /home/jupyter/models/Meta-Llama-3.1-8B-Instruct...
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:33 [gpu_model_runner.py:1985] Loading model from scratch...
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:33 [cuda.py:328] Using Flash Attention backend on V1 engine.
[1;36m(EngineCore_0 pid=82)[0;0m
Loading safetensors checkpoint shards: 0% Completed | 0/4 [00:00<?, ?it/s]
[1;36m(EngineCore_0 pid=82)[0;0m
Loading safetensors checkpoint shards: 25% Completed | 1/4 [00:03<00:10, 3.46s/it]
[1;36m(EngineCore_0 pid=82)[0;0m
...
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:10<00:00, 2.73s/it]
[1;36m(EngineCore_0 pid=82)[0;0m
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:44 [default_loader.py:262] Loading weights took 10.95 seconds
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:45 [gpu_model_runner.py:2007] Model loading took 14.9889 GiB and 11.314340 seconds
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:53 [backends.py:548] Using cache directory: /home/jupyter/.cache/vllm/torch_compile_cache/706c10d88e/rank_0_0/backbone for vLLM's torch.compile
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:53 [backends.py:559] Dynamo bytecode transform time: 8.13 s
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:02:56 [backends.py:194] Cache the graph for dynamic shape for later use
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:03:19 [backends.py:215] Compiling a graph for dynamic shape takes 25.91 s
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:03:25 [monitor.py:34] torch.compile takes 34.03 s in total
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:03:25 [gpu_worker.py:276] Available KV cache memory: 19.25 GiB
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:03:26 [kv_cache_utils.py:849] GPU KV cache size: 157,648 tokens
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:03:26 [kv_cache_utils.py:853] Maximum concurrency for 131,072 tokens per request: 1.20x
[1;36m(EngineCore_0 pid=82)[0;0m
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 0%| | 0/67 [00:00<?, ?it/s]
...
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|??????????| 67/67 [00:03<00:00, 21.97it/s]
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:03:29 [gpu_model_runner.py:2708] Graph capturing finished in 3 secs, took 0.52 GiB
[1;36m(EngineCore_0 pid=82)[0;0m INFO 08-27 21:03:29 [core.py:214] init engine (profile, create kv cache, warmup model) took 44.33 seconds
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [loggers.py:142] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 9853
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [api_server.py:1611] Supported_tasks: ['generate']
[1;36m(APIServer pid=15)[0;0m WARNING 08-27 21:03:30 [__init__.py:1625] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [serving_responses.py:120] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [serving_chat.py:134] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [serving_completion.py:77] Using default completion sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [api_server.py:1880] Starting vLLM API server 0 on http://0.0.0.0:8010
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:36] Available routes are:
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /openapi.json, Methods: GET, HEAD
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /docs, Methods: GET, HEAD
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /docs/oauth2-redirect, Methods: GET, HEAD
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /redoc, Methods: GET, HEAD
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /health, Methods: GET
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /load, Methods: GET
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /ping, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /ping, Methods: GET
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /tokenize, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /detokenize, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/models, Methods: GET
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /version, Methods: GET
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/responses, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/responses/{response_id}, Methods: GET
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/responses/{response_id}/cancel, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/chat/completions, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/completions, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/embeddings, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /pooling, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /classify, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /score, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/score, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/audio/transcriptions, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/audio/translations, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /rerank, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v1/rerank, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /v2/rerank, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /scale_elastic_ep, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /is_scaling_elastic_ep, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /invocations, Methods: POST
[1;36m(APIServer pid=15)[0;0m INFO 08-27 21:03:30 [launcher.py:44] Route: /metrics, Methods: GET
[1;36m(APIServer pid=15)[0;0m INFO: Started server process [15]
[1;36m(APIServer pid=15)[0;0m INFO: Waiting for application startup.
[1;36m(APIServer pid=15)[0;0m INFO: Application startup complete.

```




# vLLM Request Processing Workflow

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


# LLM Class

## Class Overview

The `LLM` class is a **high-level wrapper** that provides a simple Python API for running inference with large language models. It's designed for **offline/batch processing** rather than online serving.

## Key Components

### 1. **Constructor (`__init__`)**
```python
def __init__(self, model: str, *, runner: RunnerOption = "auto", ...)
```
- **Creates an `LLMEngine`** (the core inference engine) from configuration parameters
- **Sets up tokenizer, model configuration, parallel execution settings**
- **Initializes memory management** (GPU memory utilization, KV cache, etc.)
- **Handles distributed execution** (tensor/pipeline parallelism)

### 2. **Core Methods**

#### **`generate()` Method**
The main text generation method:
```python
def generate(self, prompts, sampling_params=None, *, use_tqdm=True, ...)
```
- **Validates inputs** and runner type
- **Applies LoRA requests** for fine-tuned models
- **Calls `_validate_and_add_requests()`** to process prompts
- **Runs the engine** via `_run_engine()`
- **Returns `RequestOutput` objects** with generated text

#### **`classify()` Method**
```python
def classify(self, prompts, *, use_tqdm=True, pooling_params=None, ...)
```
- **Generates classification logits** for input prompts
- **Validates that the model supports classification** (`"classify"` in supported tasks)
- **Uses pooling operations** instead of text generation
- **Returns `ClassificationRequestOutput` objects**

### 3. **Multi-Modal and Specialized Methods**

The class supports various tasks beyond text generation:

- **`embed()`**: Generate embeddings
- **`encode()`**: Generic pooling operations  
- **`reward()`**: Generate reward scores (for RLHF)
- **`score()`**: Compute similarity scores between text pairs
- **`chat()`**: Chat completion with conversation formatting
- **`beam_search()`**: Beam search generation

### 4. **Core Engine Interface**

#### **`_run_engine()` Method**
This is the **execution loop**:
```python
def _run_engine(self, *, use_tqdm=True):
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()  # Core inference step
        # Process outputs and update progress
    return sorted(outputs, key=lambda x: int(x.request_id))
```

**What happens in each iteration:**
1. **Calls `llm_engine.step()`** - runs scheduler → executor → model inference
2. **Collects finished outputs**
3. **Updates progress bar** with token speeds
4. **Continues until all requests are complete**

### 5. **Request Processing Pipeline**

#### **`_validate_and_add_requests()`**
1. **Validates prompt and parameter lengths**
2. **Sets output mode** to `FINAL_ONLY` (no streaming)
3. **Adds each request** to the engine via `_add_request()`

#### **`_add_request()`**
1. **Generates unique request ID**
2. **Calls `llm_engine.add_request()`**
3. **Passes to the scheduling layer**

## What's Special About This Class

### **Intelligent Batching**
- **Automatically batches prompts** for optimal GPU utilization
- **Handles memory constraints** dynamically
- **Supports mixed prompt lengths** efficiently

### **Multi-Task Support**
The class can run different types of tasks:
- **Generative**: Text completion, chat
- **Pooling**: Embeddings, classification, scoring
- **Multi-modal**: Images, audio, video processing

### **Progress Tracking**
- **Real-time token speed calculation**
- **Progress bars** with input/output token rates
- **Request completion tracking**

## **`classify()` method**, which:

1. **Validates the model supports classification**
2. **Calls the generic `encode()` method** with `pooling_task="classify"`
3. **Converts outputs** to `ClassificationRequestOutput` format
4. **Returns class logits** for each input prompt

This method is used for tasks like **sentiment analysis, topic classification, or any scenario where you need class probabilities** rather than generated text.

## Key Workflow Summary

```python
# User calls
llm = LLM(model="my-model")
outputs = llm.generate(["Hello world"], sampling_params)

# Internally:
# 1. LLM.__init__ → creates LLMEngine
# 2. generate() → validates inputs
# 3. _validate_and_add_requests() → processes prompts
# 4. _run_engine() → runs inference loop
# 5. llm_engine.step() → scheduler + executor + model
# 6. Returns RequestOutput objects
```

The `LLM` class essentially provides a **clean, synchronous Python API** that hides the complexity of vLLM's internal scheduling, batching, and distributed execution while providing powerful features like continuous batching and memory-efficient attention.


# `async_llm_engine.py`

The `async_llm_engine.py` file contains the **`AsyncLLMEngine`** class, which is the **asynchronous wrapper** around the synchronous `LLMEngine`. This is the component that powers vLLM's **online serving capabilities** and enables **concurrent request processing**. Let me break down the key components:

## Core Architecture Overview

The AsyncLLMEngine operates on a **producer-consumer pattern** with these main components:

1. **Background Engine Loop** - Continuously processes requests
2. **Request Tracker** - Manages incoming and ongoing requests  
3. **Async Streams** - Handles streaming outputs back to clients
4. **Request Processing Pipeline** - Coordinates with the underlying LLMEngine

## Key Classes and Components

### 1. **`AsyncStream` Class**
```python
class AsyncStream:
    """A stream of RequestOutputs for a request that can be iterated over asynchronously"""
```

**Purpose**: Represents an individual request's output stream
- **`put()`**: Adds outputs to the stream
- **`finish()`**: Marks the stream as complete
- **`generator()`**: Async generator that yields outputs to the client
- **Handles cancellation** when clients disconnect

### 2. **`RequestTracker` Class**
```python
class RequestTracker:
    """Synchronous abstraction for tracking requests."""
```

**Core Functionality**:
- **Tracks active requests** via `_request_streams` dictionary
- **Queues new requests** in `_new_requests` 
- **Queues aborted requests** in `_aborted_requests`
- **Coordinates between async client code and background loop**

**Key Methods**:
- **`add_request()`**: Adds new request to be processed
- **`abort_request()`**: Cancels a running request
- **`process_request_output()`**: Routes outputs to correct stream
- **`get_new_and_aborted_requests()`**: Retrieves requests for processing

### 3. **`_AsyncLLMEngine` Class**
**Internal async wrapper** around the synchronous `LLMEngine`:
- **`add_request_async()`**: Async version of adding requests
- **`step_async()`**: Async version of engine step
- **Handles preprocessing** and validation asynchronously

### 4. **`AsyncLLMEngine` Class**
**Main public interface** for async inference.

## Core Workflow

### **Background Engine Loop** (`run_engine_loop`)

This is the **heart of the async engine**:

```python
@staticmethod
async def run_engine_loop(engine_ref: ReferenceType):
    while True:
        if not any(has_requests_in_progress):
            # Wait for new requests
            await request_tracker.wait_for_new_requests()
            
        # Process requests for each virtual engine (pipeline parallel)
        requests_in_progress = [
            asyncio.create_task(engine.engine_step(ve))
            for ve in range(pipeline_parallel_size)
        ]
        
        # Wait for at least one virtual engine to complete
        done, _ = await asyncio.wait(requests_in_progress, 
                                   return_when=asyncio.FIRST_COMPLETED)
        
        # Handle completed virtual engines
        for task in done:
            # Schedule next step if more work exists
```

**Key Features**:
1. **Continuous Processing**: Never stops, always ready for new requests
2. **Pipeline Parallel Support**: Handles multiple virtual engines simultaneously
3. **Efficient Waiting**: Only processes when there are requests
4. **Graceful Cleanup**: Uses weak references to prevent memory leaks

### **Request Processing Flow**

#### **1. Adding Requests** (`add_request`)
```python
async def add_request(self, request_id: str, prompt: PromptType, ...):
    # Add to request tracker
    stream = self._request_tracker.add_request(request_id, ...)
    # Return async generator for client
    return stream.generator()
```

#### **2. Engine Step** (`engine_step`)
```python
async def engine_step(self, virtual_engine: int) -> bool:
    # Get new and aborted requests
    new_requests, aborted_requests = (
        self._request_tracker.get_new_and_aborted_requests())
    
    # Add new requests to engine
    for new_request in new_requests:
        await self.engine.add_request_async(**new_request)
    
    # Abort requests if needed
    if aborted_requests:
        await self._engine_abort(aborted_requests)
    
    # Execute one step of the engine
    request_outputs = await self.engine.step_async(virtual_engine)
    
    # Route outputs to streams
    self.process_request_outputs(request_outputs)
```

#### **3. Output Processing** (`process_request_outputs`)
- **Routes outputs** to the correct `AsyncStream`
- **Handles completion** and cleanup
- **Manages streaming** vs final outputs

## Key Methods

### **`generate()` Method** 
**Main text generation interface**:
```python
async def generate(self, prompt, sampling_params, request_id, ...):
    async for output in await self.add_request(...):
        yield LLMEngine.validate_output(output, RequestOutput)
```

**Features**:
- **Async generator**: Yields partial results as they're generated
- **Automatic batching**: Multiple concurrent requests processed together
- **Error handling**: Graceful cleanup on cancellation
- **Streaming support**: Real-time output delivery

### **`encode()` Method**
**For embedding and pooling tasks**:
- Similar to `generate()` but for non-generative tasks
- Returns `PoolingRequestOutput` instead of `RequestOutput`

## Advanced Features

### **1. Request Cancellation**
- **Client disconnection handling**
- **Graceful abort** of in-progress requests
- **Resource cleanup**

### **2. Concurrent Processing** 
- **Multiple requests** processed simultaneously
- **Efficient batching** across requests
- **Pipeline parallelism** support

### **3. Error Handling**
- **Background loop monitoring**
- **Dead engine detection**
- **Exception propagation** to clients

### **4. Resource Management**
- **Weak references** to prevent memory leaks
- **Automatic cleanup** of finished requests
- **Engine lifecycle management**

## Integration with API Server

The AsyncLLMEngine is primarily used by:
- **OpenAI API Server** (`vllm/entrypoints/openai/api_server.py`)
- **Custom async applications**
- **Any scenario requiring concurrent request handling**

## Key Differences from Sync LLM

| Aspect | Sync LLM | Async LLM |
|--------|-----------|-----------|
| **Usage** | Batch processing | Online serving |
| **Concurrency** | Sequential | Concurrent |
| **Streaming** | All-at-once | Real-time |
| **Client Model** | Blocking | Non-blocking |
| **Resource Sharing** | Single request | Multiple requests |

## Summary

The `AsyncLLMEngine` transforms vLLM from a **batch processing system** into a **high-performance online serving engine** by:

1. **Wrapping the synchronous engine** in an async interface
2. **Managing concurrent requests** through sophisticated request tracking
3. **Providing streaming outputs** via async generators
4. **Enabling real-time processing** with continuous background loops
5. **Supporting advanced features** like request cancellation and pipeline parallelism

This architecture enables vLLM to achieve **industry-leading serving performance** for production LLM deployments.
# AsyncLLMEngine: High-Level Design and Architecture

The `async_llm_engine.py` file implements vLLM's **asynchronous inference engine** that transforms the synchronous `LLMEngine` into a high-performance, concurrent serving system. Here are the key design concepts and patterns:

## Core Design Philosophy

The AsyncLLMEngine follows a **producer-consumer pattern** where:
- **Clients produce requests** asynchronously 
- **Background loop consumes requests** and processes them through the synchronous engine
- **Results are streamed back** to clients via async generators

## Key Components and Design Patterns

### 1. **AsyncStream - Individual Request Streaming**

```python
class AsyncStream:
    """A stream of RequestOutputs for a request that can be iterated over asynchronously"""
    
    def __init__(self, request_id: str, cancel: Callable[[str], None]):
        self.request_id = request_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    async def generator(self) -> AsyncGenerator[Union[RequestOutput, PoolingRequestOutput], None]:
        try:
            while True:
                result = await self._queue.get()
                if self._is_raisable(result):
                    if result == STOP_ITERATION:
                        return
                    raise result
                yield result
        except GeneratorExit:
            self._cancel(self.request_id)
            raise asyncio.CancelledError from None
```

**Design Concept**: Each request gets its own async stream that:
- **Queues partial results** as they become available
- **Yields results incrementally** for streaming responses
- **Handles cancellation** when clients disconnect
- **Propagates errors** appropriately

### 2. **RequestTracker - Async Request Coordination**

```python
class RequestTracker:
    """Synchronous abstraction for tracking requests."""
    
    def __init__(self):
        self._request_streams: Dict[str, AsyncStream] = {}
        self._aborted_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream, dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()
```

**Design Pattern**: **Command Queue Pattern**
- **Separates request submission from processing**
- **Decouples async client code from sync engine**
- **Provides thread-safe communication** between async and sync contexts

Key methods demonstrate the pattern:

```python
def add_request(self, request_id: str, **engine_add_request_kwargs) -> AsyncStream:
    """Add a request to be sent to the engine on the next background loop iteration."""
    stream = AsyncStream(request_id, abort_request)
    self._new_requests.put_nowait((stream, {
        "request_id": request_id,
        **engine_add_request_kwargs
    }))
    self.new_requests_event.set()
    return stream

def get_new_and_aborted_requests(self) -> Tuple[List[Dict], Set[str]]:
    """Get the new requests and finished requests to be sent to the engine."""
    # Batch process queued requests
    new_requests = []
    finished_requests = set()
    # ... batching logic
    return new_requests, finished_requests
```

### 3. **Background Engine Loop - The Heart of Async Processing**

```python
@staticmethod
async def run_engine_loop(engine_ref: ReferenceType):
    """The main async processing loop that never stops"""
    engine = engine_ref()
    if not engine:
        return

    pipeline_parallel_size = engine.engine.parallel_config.pipeline_parallel_size
    has_requests_in_progress = [False] * pipeline_parallel_size
    
    while True:
        if not any(has_requests_in_progress):
            # Wait for new requests when idle
            await engine.engine.stop_remote_worker_execution_loop_async()
            await request_tracker.wait_for_new_requests()
            
            # Create tasks for each virtual engine
            requests_in_progress = [
                asyncio.create_task(engine.engine_step(ve))
                for ve in range(pipeline_parallel_size)
            ]
            has_requests_in_progress = [True] * pipeline_parallel_size

        # Wait for at least one virtual engine to complete
        async with asyncio_timeout(ENGINE_ITERATION_TIMEOUT_S):
            done, _ = await asyncio.wait(
                requests_in_progress,
                return_when=asyncio.FIRST_COMPLETED)
        
        # Handle completed virtual engines
        for task in done:
            result = task.result()
            virtual_engine = requests_in_progress.index(task)
            # Schedule next step if more work exists
            if result or engine.engine.has_unfinished_requests_for_virtual_engine(virtual_engine):
                requests_in_progress[virtual_engine] = asyncio.create_task(
                    engine.engine_step(virtual_engine))
            else:
                has_requests_in_progress[virtual_engine] = False
```

**Design Concepts**:
- **Event-driven processing**: Only processes when requests exist
- **Pipeline parallelism support**: Handles multiple virtual engines concurrently
- **Graceful resource management**: Stops workers when idle to prevent timeouts
- **Weak references**: Prevents memory leaks from circular references

### 4. **Engine Step Processing**

```python
async def engine_step(self, virtual_engine: int) -> bool:
    """Kick the engine to process the waiting requests."""
    
    # Get batched requests from tracker
    new_requests, aborted_requests = (
        self._request_tracker.get_new_and_aborted_requests())

    # Add new requests to engine
    for new_request in new_requests:
        try:
            await self.engine.add_request_async(**new_request)
        except ValueError as e:
            self._request_tracker.process_exception(
                new_request["request_id"], e)

    # Handle aborted requests
    if aborted_requests:
        await self._engine_abort(aborted_requests)

    # Execute one step of the engine
    request_outputs = await self.engine.step_async(virtual_engine)

    # Route outputs to appropriate streams
    if not self.use_process_request_outputs_callback:
        all_finished = self.process_request_outputs(request_outputs)
    else:
        all_finished = all(request_output.finished for request_output in request_outputs)

    return not all_finished
```

**Design Pattern**: **Batch Processing Pattern**
- **Collects multiple requests** before processing
- **Efficiently processes batches** through the sync engine
- **Routes results back** to individual streams

### 5. **Multi-Engine Architecture Support**

The async engine supports multiple deployment architectures:

```python
async def build_async_engine_client_from_engine_args(engine_args, ...):
    """Create EngineClient, either in-process or multiprocess"""
    
    # V1 AsyncLLM (New architecture)
    if envs.VLLM_USE_V1:
        from vllm.v1.engine.async_llm import AsyncLLM
        async_llm = AsyncLLM.from_vllm_config(vllm_config, ...)
        yield async_llm
        
    # V0 Direct AsyncLLMEngine (Single process)
    elif disable_frontend_multiprocessing:
        engine_client = AsyncLLMEngine.from_vllm_config(vllm_config, ...)
        yield engine_client
        
    # V0 Multiprocessing Engine (Production default)
    else:
        # Spawn separate process for engine
        engine_process = context.Process(target=run_mp_engine, ...)
        # Create client to communicate with engine process
        client = MQLLMEngineClient(ipc_path, ...)
        yield client
```

### 6. **Streaming Response Generation**

```python
async def generate(self, prompt: PromptType, sampling_params: SamplingParams, 
                  request_id: str, **kwargs) -> AsyncGenerator[RequestOutput, None]:
    """Generate outputs for a request with real-time streaming"""
    
    try:
        async for output in await self.add_request(
                request_id, prompt, sampling_params, **kwargs):
            yield LLMEngine.validate_output(output, RequestOutput)
    except asyncio.CancelledError:
        await self.abort(request_id)
        raise
```

**Design Pattern**: **Async Generator Pattern**
- **Yields partial results** as they become available
- **Handles client disconnection** gracefully
- **Provides backpressure** naturally through async iteration

### 7. **Error Handling and Resilience**

```python
def _error_callback(self, exc: Exception) -> None:
    self.set_errored(exc)
    self._request_tracker.propagate_exception(exc)

async def check_health(self) -> None:
    """Raises an error if engine is unhealthy."""
    if self.is_stopped:
        raise AsyncEngineDeadError("Background loop is stopped.")
    await self.engine.check_health_async()
```

**Design Concepts**:
- **Fail-fast error propagation**: Errors immediately propagate to all streams
- **Health monitoring**: Continuous health checks prevent hanging requests
- **Graceful degradation**: Engine can recover from individual request failures

## Advanced Design Patterns

### **Lifecycle Management**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Setup: Start background tasks
        if app.state.log_stats:
            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
        
        # Optimize GC for long-running processes
        gc.collect()
        gc.freeze()
        
        yield
    finally:
        # Cleanup: Cancel tasks and cleanup state
        if task is not None:
            task.cancel()
        del app.state
```

### **Resource Optimization**

```python
def start_background_loop(self) -> None:
    """Start the background loop with proper resource management"""
    # Initialize RequestTracker with correct event loop
    self._request_tracker = RequestTracker()
    
    # Create shielded task to prevent accidental cancellation
    self._background_loop_unshielded = asyncio.get_event_loop().create_task(
        self.run_engine_loop(weakref.ref(self)))
    self.background_loop = asyncio.shield(self._background_loop_unshielded)
```

## Key Architectural Benefits

### **1. Concurrency Without Blocking**
- **Multiple requests processed simultaneously**
- **Non-blocking I/O** for client connections
- **Efficient resource utilization**

### **2. Streaming and Real-time Responses**
- **Server-Sent Events** for real-time updates
- **Backpressure handling** through async generators
- **Client disconnection detection**

### **3. Scalability and Performance**
- **Batched request processing** for efficiency
- **Pipeline parallelism** support
- **Memory-efficient streaming**

### **4. Reliability and Error Handling**
- **Graceful error propagation**
- **Request cancellation** support
- **Health monitoring** and recovery

## Integration Points

The AsyncLLMEngine serves as the **bridge** between:
- **Web frameworks** (FastAPI) and the sync engine
- **Client libraries** and vLLM's inference pipeline
- **Monitoring systems** and engine metrics
- **Load balancers** and distributed deployments

This design makes vLLM suitable for **production serving environments** where high concurrency, low latency, and reliability are critical requirements.


# `api_server.py`

The `api_server.py` file is vLLM's **OpenAI-compatible REST API server** - it's the main entry point for serving vLLM models via HTTP/REST endpoints that are compatible with OpenAI's API. Here's what's happening:

## Core Architecture

### **FastAPI-Based Web Server**
- **Framework**: Built on FastAPI for high-performance async web serving
- **OpenAI Compatibility**: Provides REST endpoints that mirror OpenAI's API structure
- **Streaming Support**: Handles both streaming and non-streaming responses
- **CORS & Security**: Includes CORS middleware, API key authentication, and SSL support

## Key Components

### 1. **Engine Client Management**

#### **Multi-Backend Support**:
```python
async def build_async_engine_client_from_engine_args(...):
    # V1 AsyncLLM (New architecture)
    if envs.VLLM_USE_V1:
        from vllm.v1.engine.async_llm import AsyncLLM
        async_llm = AsyncLLM.from_vllm_config(...)
        
    # V0 Direct AsyncLLMEngine (Single process)
    elif disable_frontend_multiprocessing:
        engine_client = AsyncLLMEngine.from_vllm_config(...)
        
    # V0 Multiprocessing Engine (Production default)
    else:
        # Spawn separate process for engine
        engine_process = context.Process(target=run_mp_engine, ...)
        # Create client to communicate with engine process
        client = MQLLMEngineClient(ipc_path, ...)
```

**Three deployment modes**:
1. **V1 Engine**: New architecture with better performance
2. **V0 Single Process**: Engine in same process (development)
3. **V0 Multiprocessing**: Engine in separate process (production default)

### 2. **Serving Handler Classes**

**Dependency injection pattern** - each endpoint gets appropriate handler:
```python
def chat(request: Request) -> Optional[OpenAIServingChat]:
    return request.app.state.openai_serving_chat

def completion(request: Request) -> Optional[OpenAIServingCompletion]:  
    return request.app.state.openai_serving_completion

def embedding(request: Request) -> Optional[OpenAIServingEmbedding]:
    return request.app.state.openai_serving_embedding
```

**Handler Types**:
- **`OpenAIServingChat`**: Chat completions (`/v1/chat/completions`)
- **`OpenAIServingCompletion`**: Text completions (`/v1/completions`)
- **`OpenAIServingEmbedding`**: Embeddings (`/v1/embeddings`)
- **`OpenAIServingTokenization`**: Tokenization utilities
- **`ServingClassification`**: Classification tasks
- **`ServingScores`**: Scoring/ranking tasks

### 3. **Main API Endpoints**

#### **Chat Completions** (`/v1/chat/completions`)
```python
@router.post("/v1/chat/completions")
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = chat(raw_request)
    generator = await handler.create_chat_completion(request, raw_request)
    
    # Handle different response types
    if isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())
    return StreamingResponse(content=generator, media_type="text/event-stream")
```

#### **Text Completions** (`/v1/completions`)
```python
@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    # Similar pattern to chat completions
```

#### **Embeddings** (`/v1/embeddings`)
```python
@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    # Returns embedding vectors for input text
```

### 4. **Additional Endpoints**

**Utility Endpoints**:
- **`/health`**: Health checks
- **`/v1/models`**: List available models
- **`/tokenize`**, **`/detokenize`**: Token utilities
- **`/version`**: vLLM version info
- **`/metrics`**: Prometheus metrics

**Advanced Features**:
- **`/classify`**: Classification tasks
- **`/score`**: Text similarity scoring
- **`/pooling`**: Generic pooling operations
- **`/rerank`**: Document reranking

### 5. **Request Processing Decorators**

#### **`@with_cancellation`**
- **Handles client disconnections** gracefully
- **Cancels ongoing requests** when clients disconnect

#### **`@load_aware_call`**
- **Load balancing** and request routing
- **Performance monitoring**

### 6. **Application Lifecycle**

#### **App Building** (`build_app()`)
```python
def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)  # Add all endpoints
    
    # Add middleware
    app.add_middleware(CORSMiddleware, ...)
    
    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(...)
```

#### **Server Startup** (`run_server_worker()`)
```python
async def run_server_worker(...):
    # 1. Create engine client
    async with build_async_engine_client(args) as engine_client:
        # 2. Build FastAPI app  
        app = build_app(args)
        
        # 3. Initialize app state with handlers
        await init_app_state(engine_client, vllm_config, app.state, args)
        
        # 4. Start HTTP server
        await serve_http(app, ...)
```

### 7. **Key Features**

#### **Streaming Support**
- **Server-Sent Events (SSE)** for real-time response streaming
- **Chunk-based processing** for long responses
- **Client disconnection handling**

#### **Multi-Modal Support**
- **Image, audio, video processing** through specialized handlers
- **File upload handling**
- **Multi-modal request routing**

#### **Authentication & Security**
- **API key validation**
- **CORS configuration**  
- **SSL/TLS support**
- **Request validation**

#### **Monitoring & Observability**
- **Prometheus metrics** integration
- **Request/response logging**
- **Health check endpoints**
- **Performance instrumentation**

## Request Flow Example

```python
# 1. Client sends request to /v1/chat/completions
# 2. FastAPI routes to create_chat_completion()
# 3. Get OpenAIServingChat handler from app state
# 4. Handler calls AsyncLLMEngine to process request
# 5. Engine returns async generator for streaming
# 6. Response streamed back to client via SSE
```

## Integration Points

**Connects To**:
- **`AsyncLLMEngine`**: Core inference engine
- **OpenAI Protocol**: Request/response schemas
- **Serving Classes**: Specialized request handlers
- **Multiprocessing Engine**: For production deployments

## Production Features

**Scalability**:
- **Multiprocessing backend** for isolation
- **Async request handling** for concurrency
- **Load balancing** support
- **Health monitoring**

**Reliability**:
- **Graceful shutdown** handling
- **Error propagation** and formatting
- **Request cancellation** support
- **Resource cleanup**

This API server transforms vLLM from a Python library into a **production-ready web service** that's compatible with existing OpenAI client libraries and tools, making it easy to drop vLLM into existing AI applications.


# `llm_engine.py`

The `LLMEngine` class is the core orchestrator of vLLM's inference system. Here are all the important functions categorized by their purpose:

## 1. **Engine Lifecycle & Initialization**

### **`__init__()`** 
```python
def __init__(self, vllm_config: VllmConfig, executor_class: Type[ExecutorBase], ...)
```
**Purpose**: Initializes the LLM engine with all necessary components
- **Sets up configurations** (model, parallel, scheduler, cache)
- **Initializes tokenizer** and input preprocessor
- **Creates schedulers** (one per virtual engine for pipeline parallelism)
- **Initializes model executor** for distributed execution
- **Sets up caches** (KV cache, multimodal cache)

### **Factory Methods**
#### **`from_vllm_config()`**  / **`from_engine_args()`** 
```python
@classmethod
def from_vllm_config(cls, vllm_config: VllmConfig, ...)
```
**Purpose**: Create engine instances from configuration objects
- **Auto-selects V0 vs V1** engine architecture
- **Handles usage context** and logging setup
- **Primary way to instantiate engines**

### **`_get_executor_cls()`**
```python
@classmethod 
def _get_executor_cls(cls, engine_config: VllmConfig) -> Type[ExecutorBase]
```
**Purpose**: Determines the appropriate executor class for the configuration
- **Chooses between UniProc, Ray, MP executors** based on parallelism settings
- **Critical for distributed execution setup**

## 2. **Request Management**

### **`add_request()`**  - **CORE FUNCTION**
```python
def add_request(self, request_id: str, prompt: PromptType, 
                params: Union[SamplingParams, PoolingParams], ...)
```
**Purpose**: Main entry point for adding requests to the engine
- **Validates request parameters** (request_id, LoRA, priority)
- **Preprocesses inputs** (tokenization, multimodal data)
- **Creates sequence groups** from prompts
- **Adds to scheduler** for processing
- **Load balances** across virtual engines

### **`_add_processed_request()`** 
```python
def _add_processed_request(self, request_id: str, processed_inputs: ProcessorInputs, ...)
```
**Purpose**: Internal method that processes validated requests
- **Creates sequences** from processed inputs
- **Handles sampling vs pooling** parameters
- **Creates sequence groups** and adds to scheduler

### **`_create_sequence_group_with_sampling()`** 
```python
def _create_sequence_group_with_sampling(self, request_id: str, seq: Sequence, ...)
```
**Purpose**: Creates sequence groups for text generation tasks
- **Handles beam search** (creates multiple sequences)
- **Sets up sampling parameters**
- **Configures LoRA requests**

### **`_create_sequence_group_with_pooling()`** 
```python  
def _create_sequence_group_with_pooling(self, request_id: str, seq: Sequence, ...)
```
**Purpose**: Creates sequence groups for embedding/classification tasks
- **Single sequence per group** (no beam search)
- **Configures pooling parameters**

### **`abort_request()`** 
```python
def abort_request(self, request_id: Union[str, Iterable[str]]) -> None
```
**Purpose**: Cancels requests that are waiting or running
- **Handles both single and batch abort**
- **Frees resources** (KV cache blocks)
- **Updates scheduler state**

## 3. **Core Execution - The Heart of vLLM**

### **`step()`** - **MOST IMPORTANT FUNCTION**
```python
def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]
```
**Purpose**: The main execution loop - performs one iteration of inference

**Three-Phase Process**:

#### **Phase 1: Scheduling**
- **Calls scheduler** to decide which requests to process
- **Determines memory allocation** (KV cache blocks)
- **Handles preemption** if memory is full
- **Supports chunked prefill** for long prompts

#### **Phase 2: Model Execution**
```python
outputs = self.model_executor.execute_model(execute_model_req)
```
- **Creates ExecuteModelRequest** with all necessary data
- **Calls distributed executor** to run model forward pass
- **Handles failures** and error recovery

#### **Phase 3: Post-Processing**
- **Processes model outputs** through `_process_model_outputs()`
- **Updates sequences** with new tokens
- **Checks stopping criteria**
- **Creates RequestOutput objects**

### **`_process_model_outputs()`** 
```python
def _process_model_outputs(self, ctx: SchedulerContext, request_id: Optional[str] = None)
```
**Purpose**: Processes raw model outputs into final results
- **Handles sampling** (temperature, top-p, top-k)
- **Applies stopping criteria** (stop tokens, max length)
- **Updates sequence states**
- **Creates output objects**

### **`_advance_to_next_step()`** 
```python
def _advance_to_next_step(self, output: SamplerOutput, seq_group_metadata_list, ...)
```
**Purpose**: Updates sequences with generated tokens
- **Appends new tokens** to sequences
- **Updates logprobs** and embeddings
- **Advances sequence state**

## 4. **State Management & Queries**

### **Request State Queries**
```python
def get_num_unfinished_requests(self) -> int  
def has_unfinished_requests(self) -> bool     
def has_unfinished_requests_for_virtual_engine(self, virtual_engine: int) -> bool 
```
**Purpose**: Query engine state for request management
- **Used by async engine** to know when to wait
- **Critical for pipeline parallelism**

### **Configuration Getters**
```python
def get_vllm_config(self) -> VllmConfig        
def get_model_config(self) -> ModelConfig      
def get_parallel_config(self) -> ParallelConfig 
def get_scheduler_config(self) -> SchedulerConfig 
```
**Purpose**: Provide access to engine configurations
- **Used by API servers** and monitoring
- **Supports introspection**

## 5. **Cache Management**

### **`reset_mm_cache()`** (Line 841)
```python
def reset_mm_cache(self) -> bool
```
**Purpose**: Clears multimodal processor cache
- **Frees memory** from image/audio processing
- **Used between requests** with different modalities

### **`reset_prefix_cache()`** (Line 846)
```python  
def reset_prefix_cache(self, device: Optional[Device] = None) -> bool
```
**Purpose**: Clears prefix caching state
- **Resets cached prompt prefixes**
- **Important for memory management**

## 6. **Performance & Observability**

### **`do_log_stats()`** 
```python
def do_log_stats(self, scheduler_outputs: Optional[SchedulerOutputs] = None, ...)
```
**Purpose**: Comprehensive performance logging and metrics collection
- **System metrics**: GPU/CPU cache usage, queue sizes
- **Request metrics**: Latency, throughput, token counts
- **LoRA metrics**: Active adapter usage
- **Speculative decoding metrics**

Key metrics tracked:
- **Time to First Token (TTFT)**
- **Time Per Output Token (TPOT)**
- **End-to-end latency**
- **Cache hit rates**
- **Queue wait times**

### **`start_profile()` / `stop_profile()`** 
```python
def start_profile(self) -> None
def stop_profile(self) -> None
```
**Purpose**: Performance profiling support
- **Enables CUDA profiling**
- **Used for performance debugging**

### **Tracing Functions**
#### **`do_tracing()`** (Line 1688) / **`create_trace_span()`**
**Purpose**: OpenTelemetry tracing support
- **Creates trace spans** for requests
- **Tracks request lifecycle**
- **Supports distributed tracing**

## 7. **Utility & Validation**

### **`_validate_model_inputs()`** 
```python
def _validate_model_inputs(self, inputs: ProcessorInputs, lora_request: Optional[LoRARequest])
```
**Purpose**: Validates input prompts and tokens
- **Checks token limits**
- **Validates multimodal inputs**
- **Ensures LoRA compatibility**

### **`_has_remaining_steps()`** 
```python
def _has_remaining_steps(self, seq_group_metadata_list: Optional[List[SequenceGroupMetadata]]) -> bool
```
**Purpose**: Checks if multi-step decoding has remaining steps
- **Used in chunked prefill**
- **Supports speculative decoding**

### **Static Validation Methods**
#### **`validate_output()` / `validate_outputs()`**
```python
@staticmethod
def validate_output(output, output_type: Type[_O]) -> _O
def validate_outputs(outputs: List[Any], output_type: Type[_O]) -> List[_O]
```
**Purpose**: Type validation for outputs
- **Ensures type safety**
- **Used in testing**

## **Critical Function Interactions**

### **Main Execution Flow**:
1. **`add_request()`** → **`_add_processed_request()`** → **scheduler**
2. **`step()`** → **scheduler.schedule()** → **model_executor.execute_model()**
3. **`step()`** → **`_process_model_outputs()`** → **RequestOutput**

### **Memory Management Flow**:
- **Scheduler** manages KV cache allocation
- **`reset_*_cache()`** functions manage cache lifecycle
- **`abort_request()`** frees resources

### **Observability Flow**:
- **`step()`** → **`do_log_stats()`** → metrics
- **`step()`** → **`do_tracing()`** → traces
- **Background threads** collect and export metrics

## **Function Importance Ranking**

1. **`step()`** - The execution heart
2. **`add_request()`** - Request entry point
3. **`_process_model_outputs()`** - Output processing
4. **`from_engine_args()`** - Engine creation
5. **`do_log_stats()`** - Performance monitoring
6. **`abort_request()`** - Resource management
7. **Configuration getters** - State introspection

The `LLMEngine` is essentially a sophisticated **request orchestrator** that coordinates between the scheduler (resource allocation), executor (model execution), and output processors (result formatting) while providing comprehensive observability and error handling.


# InputPreprocessor: High-Level Design and Architecture

The `preprocess.py` file implements vLLM's **unified input preprocessing system** that transforms diverse input formats into standardized internal representations. The design is built around **polymorphism, async/sync duality, and multi-modal extensibility**.

## Core Design Philosophy

The `InputPreprocessor` class follows a **strategy pattern** combined with **factory pattern** to handle:
- **Multiple input types**: Text, tokens, embeddings, and multimodal data
- **Multiple model architectures**: Decoder-only (GPT-style) and encoder-decoder (T5-style) models  
- **Async and sync processing**: Supporting both blocking and non-blocking operations
- **Extensible multimodal support**: Images, audio, video through pluggable processors

## Key Design Patterns

### 1. **Polymorphic Input Processing**

The preprocessor handles multiple input formats through a unified interface:

```python
class InputPreprocessor:
    def __init__(self, model_config: ModelConfig, tokenizer: Optional[TokenizerGroup], 
                 mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY):
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.mm_registry = mm_registry
```

**Design Concept**: Single entry point that adapts to different input types and model architectures.

### 2. **Main Processing Entry Points**

```python
def preprocess(self, prompt: PromptType, tokenization_kwargs: Optional[dict[str, Any]] = None,
               lora_request: Optional[LoRARequest] = None) -> ProcessorInputs:
    """Preprocess the input prompt."""
    if self.model_config.is_encoder_decoder:
        # Encoder-decoder model requires special mapping
        return self._process_encoder_decoder_prompt(prompt, tokenization_kwargs)
    
    if is_explicit_encoder_decoder_prompt(prompt):
        raise ValueError("Cannot pass encoder-decoder prompt to decoder-only models")
    
    # Decoder-only operation
    return self._process_decoder_only_prompt(prompt, tokenization_kwargs, lora_request)

async def preprocess_async(self, prompt: PromptType, ...) -> ProcessorInputs:
    """Async version of preprocess with parallel processing capabilities"""
    # Similar logic but with async/await for concurrent operations
```

**Design Pattern**: **Template Method Pattern**
- **Main algorithm** is the same for sync/async
- **Implementation details** differ (sync vs async tokenization, multimodal processing)
- **Architecture branching** handles encoder-decoder vs decoder-only models

### 3. **Type-Based Input Dispatching**

```python
def _prompt_to_llm_inputs(self, prompt: SingletonPrompt, ...) -> SingletonInputs:
    """Extract the singleton inputs from a prompt."""
    parsed = parse_singleton_prompt(prompt)
    
    if parsed["type"] == "embeds":
        return self._process_embeds(parsed["content"])
    if parsed["type"] == "tokens":
        return self._process_tokens(parsed["content"], lora_request=lora_request)
    if parsed["type"] == "text":
        return self._process_text(parsed["content"], tokenization_kwargs, lora_request)
    if parsed["type"] == "str":
        return self._process_text(TextPrompt(prompt=parsed["content"]), 
                                  tokenization_kwargs, lora_request)
```

**Design Pattern**: **Visitor Pattern**
- **Input types are parsed** and dispatched to specific handlers
- **Each handler** knows how to process its specific input type
- **Extensible** for new input types

### 4. **Multimodal Processing Architecture**

```python
def _process_multimodal(self, prompt: Union[str, list[int]], mm_data: MultiModalDataDict,
                       mm_processor_kwargs: Optional[Mapping[str, object]], ...) -> MultiModalInputs:
    """Process multimodal inputs through the registry system"""
    
    # Use the multimodal registry to process different data types
    mm_inputs = self.mm_registry.process_input(
        self.model_config,
        mm_data,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    
    # Handle text prompt alongside multimodal data
    if isinstance(prompt, str):
        prompt_token_ids = self._tokenize_prompt(prompt, lora_request, tokenization_kwargs)
    else:
        prompt_token_ids = prompt
    
    return MultiModalInputs(
        type="multimodal",
        prompt=prompt if isinstance(prompt, str) else "",
        prompt_token_ids=prompt_token_ids,
        mm_kwargs=mm_inputs.mm_kwargs,
        mm_hashes=mm_inputs.mm_hashes,
        mm_placeholders=mm_inputs.mm_placeholders,
    )
```

**Design Concept**: **Registry Pattern**
- **Pluggable processors** for different modalities (image, audio, video)
- **Standardized interface** between preprocessor and multimodal handlers
- **Caching support** through hashing mechanisms

### 5. **Encoder-Decoder Model Support**

```python
def _process_encoder_decoder_prompt(self, prompt: PromptType, ...) -> EncoderDecoderInputs:
    """Special handling for encoder-decoder architectures"""
    
    if is_explicit_encoder_decoder_prompt(prompt):
        # User explicitly provided both encoder and decoder prompts
        encoder_inputs = self._prompt_to_llm_inputs(
            prompt["encoder_prompt"], tokenization_kwargs=tokenization_kwargs)
        
        if (decoder_input := prompt["decoder_prompt"]) is None:
            decoder_inputs = None
        else:
            decoder_inputs = self._prompt_to_llm_inputs(decoder_input)
            
        # Handle multimodal inputs specially for enc-dec
        if self.model_config.is_multimodal_model:
            encoder_inputs, decoder_inputs = self._split_enc_dec_mm_inputs(
                encoder_inputs, decoder_inputs)
    else:
        # Single prompt - need to determine encoder vs decoder usage
        inputs = self._prompt_to_llm_inputs(prompt, tokenization_kwargs)
        
        if self.model_config.is_multimodal_model:
            encoder_inputs, decoder_inputs = self._split_enc_dec_mm_inputs(inputs)
        else:
            encoder_inputs = inputs
            decoder_inputs = None
    
    return self._build_enc_dec_llm_inputs(encoder_inputs, decoder_inputs)
```

**Design Concepts**:
- **Adaptive prompt routing**: Single prompts can go to encoder or decoder based on model type
- **Special cases**: Whisper models route text to decoder, multimodal data to encoder
- **Default decoder generation**: Creates appropriate starting tokens when decoder prompt is missing

### 6. **Tokenization Strategy Pattern**

```python
def _get_tokenization_kw(self, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Model-specific tokenization parameters"""
    kwargs = dict[str, Any]()
    
    if self.model_config.hf_config.model_type == "whisper":
        # Whisper needs special token handling
        kwargs["add_special_tokens"] = False
    
    if overrides:
        kwargs.update(overrides)
    
    return kwargs

def _tokenize_prompt(self, prompt: str, lora_request: Optional[LoRARequest], 
                    tokenization_kwargs: Optional[dict[str, Any]] = None) -> list[int]:
    """Apply the model's tokenizer with model-specific adaptations"""
    tokenizer = self.get_tokenizer_group()
    tokenization_kwargs = self._get_tokenization_kw(tokenization_kwargs)
    
    return tokenizer.get_lora_tokenizer(lora_request).encode(prompt, **tokenization_kwargs)
```

**Design Pattern**: **Adapter Pattern**
- **Different models** require different tokenization strategies
- **LoRA support**: Different tokenizers for different LoRA adapters
- **Override mechanisms**: Allow custom tokenization parameters

### 7. **Async Processing with Concurrency**

```python
async def _process_encoder_decoder_prompt_async(self, prompt: PromptType, ...) -> EncoderDecoderInputs:
    """Async processing with parallel encoder/decoder handling"""
    
    if is_explicit_encoder_decoder_prompt(prompt):
        encoder_task = self._prompt_to_llm_inputs_async(
            prompt["encoder_prompt"], tokenization_kwargs=tokenization_kwargs)
        
        if (decoder_input := prompt["decoder_prompt"]) is None:
            encoder_inputs = await encoder_task
            decoder_inputs = None
        else:
            decoder_task = self._prompt_to_llm_inputs_async(
                decoder_input, tokenization_kwargs=tokenization_kwargs)
            
            # Process encoder and decoder in parallel
            encoder_inputs, decoder_inputs = await asyncio.gather(encoder_task, decoder_task)
```

**Design Concept**: **Parallel Processing Pattern**
- **Concurrent tokenization**: Encoder and decoder prompts processed simultaneously
- **I/O optimization**: Async tokenization prevents blocking
- **Resource efficiency**: Better utilization of tokenization resources

### 8. **Special Token Management**

```python
def get_bos_token_id(self, lora_request: Optional[LoRARequest] = None) -> Optional[int]:
    """Get beginning-of-sequence token with LoRA support"""
    if self.tokenizer is None:
        logger.warning("Using None for BOS token id because tokenizer is not initialized")
        return None
    return self.tokenizer.get_lora_tokenizer(lora_request).bos_token_id

def _get_default_enc_dec_decoder_prompt(self) -> list[int]:
    """Generate default decoder prompt for encoder-decoder models"""
    bos_token_id = self.get_bos_token_id()
    assert bos_token_id is not None
    return [bos_token_id]

def _prepare_decoder_input_ids_for_generation(self, decoder_input_ids: Optional[list[int]]) -> list[int]:
    """Prepare decoder inputs following HuggingFace conventions"""
    decoder_start_token_id = self.get_decoder_start_token_id()
    assert decoder_start_token_id is not None
    
    if decoder_input_ids is None:
        decoder_input_ids = self._get_default_enc_dec_decoder_prompt()
    
    if (len(decoder_input_ids) == 0 
        or decoder_input_ids[0] != decoder_start_token_id):
        decoder_input_ids = [decoder_start_token_id] + decoder_input_ids
    
    return decoder_input_ids
```

**Design Concepts**:
- **HuggingFace compatibility**: Follows transformers library conventions
- **Model-specific logic**: Different models have different special token requirements
- **Graceful defaults**: Automatic generation of appropriate starting sequences

## Advanced Design Features

### **Input Type Abstraction**

The system handles multiple input representations:

```python
# Text inputs
TextPrompt(prompt="Hello world")

# Token inputs  
TokensPrompt(prompt_token_ids=[1, 2, 3])

# Embedding inputs
EmbedsPrompt(prompt_embeds=tensor)

# Multimodal inputs
TextPrompt(prompt="Describe this image", multi_modal_data={"image": image_data})

# Encoder-decoder inputs
{"encoder_prompt": "English text", "decoder_prompt": "French text"}
```

### **Caching and Optimization**

```python
# Cache salt for request deduplication
if cache_salt := parsed_content.get("cache_salt"):
    inputs["cache_salt"] = cache_salt

# Multimodal hashing for efficient caching
mm_hashes=mm_inputs.mm_hashes
```

### **Error Handling and Validation**

```python
if self.tokenizer is None:
    raise ValueError("You cannot pass text prompts when `skip_tokenizer_init` is True")

if not self.model_config.is_encoder_decoder:
    logger.warning_once("Using None for decoder start token id because "
                       "this is not an encoder/decoder model.")
```

## Integration Points

The `InputPreprocessor` serves as the **universal adapter** between:
- **Raw user inputs** (text, images, tokens) and internal representations
- **Different model architectures** (decoder-only vs encoder-decoder)
- **Sync and async execution** contexts
- **Multimodal processors** and the core engine
- **LoRA adapters** and tokenization systems

## Key Benefits

1. **Unified Interface**: Single entry point for all input types
2. **Architecture Agnostic**: Handles both decoder-only and encoder-decoder models
3. **Multimodal Extensibility**: Pluggable processor system
4. **Performance Optimization**: Async processing with concurrency
5. **LoRA Support**: Adaptive tokenization for fine-tuned models
6. **Caching Support**: Efficient request deduplication
7. **Error Resilience**: Graceful handling of edge cases

This design makes vLLM capable of handling diverse input formats while maintaining **type safety, performance, and extensibility** across different model architectures and use cases.

# Scheduler: High-Level Design and Architecture

The `scheduler.py` file implements vLLM's **core request scheduling and memory management system** that enables efficient continuous batching and resource allocation. The design follows a **sophisticated resource-aware scheduling pattern** that optimizes for throughput while maintaining fairness.

## Core Design Philosophy

The scheduler operates on the principle of **"maximize throughput while respecting resource constraints"** through:
- **Continuous batching**: Mix new requests with ongoing generation
- **Intelligent preemption**: Swap out requests when memory is full
- **Priority-aware scheduling**: Support different scheduling policies  
- **Memory-efficient operations**: Minimize GPU memory fragmentation

## Key Components and Design Patterns

### 1. **Three-Queue Architecture**

```python
class Scheduler:
    def __init__(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig, ...):
        # Sequence groups in different states
        self.waiting: Deque[SequenceGroup] = deque()    # New/preempted requests
        self.running: Deque[SequenceGroup] = deque()    # Currently executing
        self.swapped: Deque[SequenceGroup] = deque()    # Swapped to CPU memory
        
        # Block space manager for memory allocation
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            enable_caching=self.cache_config.enable_prefix_caching,
        )
```

**Design Pattern**: **State Machine Pattern**
- **Requests flow** between waiting → running → finished/swapped
- **State transitions** are managed based on resource availability
- **Queue-based organization** enables efficient FIFO/priority scheduling

### 2. **Resource Budget Management**

```python
@dataclass
class SchedulingBudget:
    """Resource constraints for each scheduling iteration"""
    
    token_budget: int                    # Max tokens per batch
    max_num_seqs: int                   # Max sequences per batch
    _num_batched_tokens: int = 0        # Current token usage
    _num_curr_seqs: int = 0             # Current sequence count
    
    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        """Check if new request fits within budget"""
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)
    
    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int, 
                              num_cached_tokens: int = 0):
        """Account for newly scheduled tokens"""
        if req_id in self._request_ids_num_batched_tokens:
            return  # Avoid double-counting
        self._num_batched_tokens += num_batched_tokens
        self._num_cached_tokens += num_cached_tokens
```

**Design Concept**: **Resource Accounting Pattern**
- **Fine-grained tracking** of token and sequence budgets
- **Deduplication** prevents double-counting same request
- **Cache-aware accounting** distinguishes cached vs new tokens

### 3. **Main Scheduling Algorithm**

```python
def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
    """Main scheduling entry point"""
    scheduler_start_time = time.perf_counter()
    
    # Core scheduling logic
    scheduler_outputs: SchedulerOutputs = self._schedule()
    
    # Create metadata for model execution
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
        seq_group = scheduled_seq_group.seq_group
        token_chunk_size = scheduled_seq_group.token_chunk_size
        
        # Build sequence data and block tables
        seq_data: Dict[int, SequenceData] = {}
        block_tables: Dict[int, List[int]] = {}
        
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq_data[seq.seq_id] = seq.data
            block_tables[seq.seq_id] = self.block_manager.get_block_table(seq)
            
        # Create metadata for workers
        seq_group_metadata = SequenceGroupMetadata(
            request_id=seq_group.request_id,
            is_prompt=seq_group.is_prefill(),
            seq_data=seq_data,
            sampling_params=seq_group.sampling_params,
            block_tables=block_tables,
            token_chunk_size=token_chunk_size,
            lora_request=seq_group.lora_request,
            multi_modal_data=seq_group.multi_modal_data,
        )
        seq_group_metadata_list.append(seq_group_metadata)
    
    return seq_group_metadata_list, scheduler_outputs, allow_async_output_proc
```

**Design Pattern**: **Template Method Pattern**
- **Main algorithm structure** is fixed
- **Specific scheduling policies** implemented in `_schedule()` variants
- **Metadata preparation** is consistent across policies

### 4. **Default Scheduling Policy**

```python
def _schedule_default(self) -> SchedulerOutputs:
    """Throughput-optimized scheduling policy"""
    
    # Initialize resource budget
    budget = SchedulingBudget(
        token_budget=self.scheduler_config.max_num_batched_tokens,
        max_num_seqs=self.scheduler_config.max_num_seqs,
    )
    
    # Account for currently running requests
    for seq_group in self.running:
        budget.add_num_seqs(seq_group.request_id,
                           seq_group.get_max_num_running_seqs())
    
    # Phase 1: Schedule prefill requests (new prompts)
    prefills = SchedulerPrefillOutputs.create_empty()
    if not self.swapped:  # Prioritize swapped requests
        prefills = self._schedule_prefills(budget, curr_loras, enable_chunking=False)
    
    # Phase 2: Schedule running requests (decode steps)
    running_scheduled = SchedulerRunningOutputs.create_empty()
    if len(prefills.seq_groups) == 0:  # Don't mix prefill and decode
        running_scheduled = self._schedule_running(budget, curr_loras, enable_chunking=False)
    
    # Phase 3: Swap in requests from CPU memory
    swapped_in = SchedulerSwappedInOutputs.create_empty()
    if len(running_scheduled.preempted) + len(running_scheduled.swapped_out) == 0:
        swapped_in = self._schedule_swapped(budget, curr_loras)
    
    # Update queue states
    self.waiting.extendleft(running_scheduled.preempted)
    self.running.extend([s.seq_group for s in prefills.seq_groups])
    self.running.extend(running_scheduled.decode_seq_groups_list)
    self.swapped.extend(running_scheduled.swapped_out)
```

**Design Concept**: **Phase-Based Scheduling**
- **Prefill priority**: New requests get priority for better latency
- **No mixing**: Prefill and decode are separate phases for efficiency
- **Preemption handling**: Memory pressure triggers swapping/recomputation

### 5. **Prefill Scheduling with Chunking**

```python
def _schedule_prefills(self, budget: SchedulingBudget, curr_loras: Optional[Set[int]],
                      enable_chunking: bool = False) -> SchedulerPrefillOutputs:
    """Schedule waiting requests for prefill processing"""
    
    seq_groups: List[ScheduledSequenceGroup] = []
    ignored_seq_groups: List[SequenceGroup] = []
    
    while self._passed_delay(time.time()) and self.waiting:
        seq_group = self.waiting[0]
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        
        # Calculate token requirements
        num_new_tokens_uncached, num_new_tokens_cached = (
            self._get_num_new_uncached_and_cached_tokens(
                seq_group, SequenceStatus.WAITING, enable_chunking, budget))
        num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached
        
        # Check prompt length limits
        prompt_limit = self._get_prompt_limit(seq_group)
        if num_new_tokens > prompt_limit:
            # Mark as ignored and continue
            for seq in waiting_seqs:
                seq.status = SequenceStatus.FINISHED_IGNORED
            ignored_seq_groups.append(seq_group)
            self.waiting.popleft()
            continue
        
        # Check memory allocation
        can_allocate = self.block_manager.can_allocate(seq_group)
        if can_allocate == AllocStatus.LATER:
            break  # No more memory available
        elif can_allocate == AllocStatus.NEVER:
            # Will never fit - ignore request
            ignored_seq_groups.append(seq_group)
            self.waiting.popleft()
            continue
        
        # Check budget constraints
        num_new_seqs = seq_group.get_max_num_running_seqs()
        if not budget.can_schedule(num_new_tokens=num_new_tokens,
                                 num_new_seqs=num_new_seqs):
            break  # Budget exhausted
        
        # Schedule the request
        self._allocate_and_set_running(seq_group)
        seq_groups.append(ScheduledSequenceGroup(
            seq_group=seq_group, token_chunk_size=num_new_tokens))
        budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens_uncached,
                                    num_new_tokens_cached)
        budget.add_num_seqs(seq_group.request_id, num_new_seqs)
        self.waiting.popleft()
```

**Design Features**:
- **Chunked prefill**: Large prompts can be processed in chunks
- **Memory-aware**: Checks allocation before committing
- **Budget-aware**: Respects token and sequence limits
- **Graceful degradation**: Ignores requests that won't fit

### 6. **Intelligent Preemption System**

```python
def _preempt(self, seq_group: SequenceGroup, 
            blocks_to_swap_out: List[Tuple[int, int]]) -> PreemptionMode:
    """Choose and execute preemption strategy"""
    
    # Determine preemption mode
    if self.user_specified_preemption_mode is not None:
        preemption_mode = self.user_specified_preemption_mode
    else:
        # Auto-select based on sequence characteristics
        if seq_group.get_max_num_running_seqs() == 1:
            # Single sequence: recomputation is cheaper
            preemption_mode = PreemptionMode.RECOMPUTE
        else:
            # Multiple sequences (beam search): swapping preserves work
            preemption_mode = PreemptionMode.SWAP
    
    if preemption_mode == PreemptionMode.RECOMPUTE:
        self._preempt_by_recompute(seq_group)
    elif preemption_mode == PreemptionMode.SWAP:
        self._preempt_by_swap(seq_group, blocks_to_swap_out)
    
    return preemption_mode

def _preempt_by_recompute(self, seq_group: SequenceGroup) -> None:
    """Discard KV cache and restart from beginning"""
    seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
    for seq in seqs:
        seq.status = SequenceStatus.WAITING
        self.free_seq(seq)  # Free GPU memory blocks
        seq.reset_state_for_recompute()  # Reset sequence state

def _preempt_by_swap(self, seq_group: SequenceGroup, 
                    blocks_to_swap_out: List[Tuple[int, int]]) -> None:
    """Move KV cache to CPU memory"""
    self._swap_out(seq_group, blocks_to_swap_out)
```

**Design Pattern**: **Strategy Pattern**
- **Multiple preemption strategies** based on sequence characteristics
- **Automatic selection**: Single sequences use recomputation, multiple use swapping
- **Resource optimization**: Minimizes wasted computation

### 7. **Chunked Prefill Support**

```python
def _schedule_chunked_prefill(self) -> SchedulerOutputs:
    """Advanced scheduling with chunked prefill support"""
    
    budget = SchedulingBudget(
        token_budget=self.scheduler_config.max_num_batched_tokens,
        max_num_seqs=self.scheduler_config.max_num_seqs,
    )
    
    # First, schedule already running requests (ongoing chunks)
    running_scheduled = self._schedule_running(budget, curr_loras, enable_chunking=True)
    
    # Then, try to fill remaining budget with new prefills
    prefills = SchedulerPrefillOutputs.create_empty()
    if budget.remaining_token_budget() > 0:
        prefills = self._schedule_prefills(budget, curr_loras, enable_chunking=True)
    
    # Combine prefill and decode into single batch
    scheduled_seq_groups = []
    scheduled_seq_groups.extend(running_scheduled.prefill_seq_groups)  # Chunked prefills
    scheduled_seq_groups.extend(prefills.seq_groups)                  # New prefills
    scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)  # Decodes
```

**Design Concept**: **Mixed Workload Optimization**
- **Prefill chunking**: Large prompts processed incrementally
- **Workload mixing**: Prefill chunks and decode steps in same batch
- **Budget optimization**: Fill available compute with best mix of work

### 8. **Priority Scheduling**

```python
def _schedule_priority_preemption(self, budget: SchedulingBudget) -> int:
    """Preempt low-priority requests for high-priority ones"""
    
    waiting_queue = deque(self.waiting)
    running_queue = deque(self.running)
    
    force_preemption_count = 0
    while waiting_queue:
        seq_group = waiting_queue.popleft()
        
        # Try to fit within budget
        num_new_tokens = self._get_num_new_tokens(seq_group)
        if budget.can_schedule(num_new_tokens=num_new_tokens, num_new_seqs=1):
            break  # Found space naturally
        
        # Need to preempt - find victim with lower priority
        while running_queue and force_preemption_count < self.max_num_running_seqs:
            if running_queue[-1].priority >= seq_group.priority:
                break  # No suitable victim
            
            # Preempt the victim
            victim = running_queue.pop()
            self._preempt(victim, blocks_to_swap_out)
            waiting_queue.appendleft(victim)
            force_preemption_count += 1
        
        # Re-sort waiting queue by priority
        waiting_queue = deque(sorted(waiting_queue, key=self._get_priority))
    
    return force_preemption_count
```

## Advanced Design Features

### **Prefix Caching Integration**

```python
# Cache-aware token counting
if self.cache_config.enable_prefix_caching:
    common_computed_block_nums = (
        self.block_manager.get_common_computed_block_ids(
            seq_group.get_seqs(status=SequenceStatus.RUNNING)))
```

### **LoRA Batching Optimization**

```python
# Efficient LoRA batching
curr_loras = set(seq_group.lora_int_id for seq_group in self.running 
                if seq_group.lora_int_id > 0)

# Sort scheduled groups by LoRA ID for efficient execution
def _sort_by_lora_ids(self):
    def key_fn(group: ScheduledSequenceGroup):
        return (group.seq_group.lora_int_id, group.seq_group.request_id)
    self.scheduled_seq_groups = sorted(self.scheduled_seq_groups, key=key_fn)
```

### **Memory-Efficient Object Pooling**

```python
# Reuse expensive objects across iterations
self._seq_group_metadata_cache: List[PyObjectCache] = []
self._scheduler_running_outputs_cache: List[PyObjectCache] = []

# Get cached object instead of allocating new
seq_group_metadata = self._seq_group_metadata_cache[self.cache_id].get_object()
```

## Key Design Benefits

1. **High Throughput**: Continuous batching maximizes GPU utilization
2. **Low Latency**: Prefill priority and chunking reduce time-to-first-token
3. **Memory Efficiency**: Intelligent preemption and prefix caching
4. **Fairness**: Priority scheduling and anti-starvation mechanisms
5. **Scalability**: Resource-aware scheduling adapts to available memory
6. **Flexibility**: Multiple scheduling policies for different workloads

The scheduler serves as vLLM's **traffic controller**, intelligently managing the flow of requests through the system while maximizing resource utilization and maintaining quality of service guarantees. Its sophisticated design enables vLLM to achieve industry-leading throughput and latency characteristics.
