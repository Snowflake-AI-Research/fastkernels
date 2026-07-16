# End-to-End Serving Framework

FastKernels isn't just for benchmarking; it is also a minimal, high-performance end-to-end serving framework. You can use it to run models efficiently for inference.

You can start the serving engine directly from the CLI. 

```bash
# Serve a model natively
fastkernels serve --model meta-llama/Llama-3.1-8B-Instruct --tp 1
```
Once the engine is running, you can send requests to it using standard HTTP calls (compatible with OpenAI's API format), allowing easy integration into existing applications.

## How It Works

Because FastKernels dynamically swaps candidate implementations into the L4 architecture pipelines, any optimized kernels you've developed are automatically used when serving. The serving engine orchestrates token generation, batching, and context management across your GPUs.

This provides a drop-in inference endpoint that benefits immediately from all optimized L1-L3 operators you have developed.
