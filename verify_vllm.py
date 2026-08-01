if __name__ == '__main__':
    from vllm import LLM, SamplingParams

    prompts = [
        "Who are you?",
        "What is the capital of France?",
        "What is the largest planet in our solar system?"
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="Qwen/Qwen3Guard-Gen-0.6B", gpu_memory_utilization=0.8, max_model_len=16096)

    # llm = LLM(model="meta-llama/Llama-3.1-8B", gpu_memory_utilization=0.8, max_model_len=16096)
    # llm = LLM(model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", gpu_memory_utilization=0.8, max_model_len=16096)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated Text: {generated_text!r}")