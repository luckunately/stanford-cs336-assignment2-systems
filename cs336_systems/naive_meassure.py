import statistics
from urllib import parse
from cs336_basics import model
import torch
import argparse
import timeit


def parse_args():
    parser = argparse.ArgumentParser(description="Naive Measurement Script")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="The number of unique items in the output vocabulary to be predicted.",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="The maximum number of tokens to process at once.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="The dimensionality of the model embeddings and sublayer outputs.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="The number of Transformer layers to use.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of heads to use in multi-headed attention. d_model must be evenly divisible by num_heads.",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=2048,
        help="Dimensionality of the feed-forward inner layer.",
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=10000.0,
        help="The theta value for the RoPE positional encoding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The number of sequences to process in parallel during training.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="The number of warmup steps for the learning rate scheduler.",
    )
    parser.add_argument(
        "--measure_steps",
        type=int,
        default=10,
        help="The number of steps to measure execution time over.",
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Enable automatic mixed precision for faster computation on compatible hardware.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile for potential performance improvements.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_instance = model.BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    if args.torch_compile:
        model_instance = torch.compile(model_instance)
        # make sure it is still a nn.Module
        assert isinstance(
            model_instance, torch.nn.Module
        ), "Compiled model is not an instance of torch.nn.Module"

    dummy_input = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length)
    )

    if args.autocast:
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        ):
            # Warm-up
            warmup(model_instance, dummy_input, args.warmup_steps)

            # Measure execution time and backward time separately
            inference_times, backward_times = test_area(
                model_instance, dummy_input, args.measure_steps
            )
    else:
        # Warm-up
        warmup(model_instance, dummy_input, args.warmup_steps)

        # Measure execution time and backward time separately
        inference_times, backward_times = test_area(
            model_instance, dummy_input, args.measure_steps
        )

    print("\n==== Summary ====")
    print(
        f"Average Inference Time: {sum(inference_times) / len(inference_times):.6f} seconds"
    )
    print(
        f"Average Backward Time: {sum(backward_times) / len(backward_times):.6f} seconds"
    )

    # calculate standard deviation and then make a table

    inf_stddev = statistics.stdev(inference_times)
    back_stddev = statistics.stdev(backward_times)
    print(f"Inference Time Stddev: {inf_stddev:.6f} seconds")
    print(f"Backward Time Stddev: {back_stddev:.6f} seconds")
    print("\nDetailed Timing Table:")
    print("Step\tInference Time (s)\tBackward Time (s)")
    for i, (inf_time, back_time) in enumerate(zip(inference_times, backward_times)):
        print(f"{i+1}\t{inf_time:.6f}\t\t{back_time:.6f}")
    # For example: ddp_model.reset_gradient_synchronization()


def warmup(model: torch.nn.Module, input: torch.Tensor, steps: int):
    for _ in range(steps):
        out = model(input)
        _ = out.sum().backward()
    torch.cuda.synchronize() if torch.cuda.is_available() else None


def test_area(model: torch.nn.Module, input: torch.Tensor, steps: int):
    inference_times = []
    backward_times = []
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(steps):
        start_time = timeit.default_timer()
        out = model(input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = timeit.default_timer() - start_time
        inference_times.append(inference_time)

        start_time = timeit.default_timer()
        _ = out.sum().backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_time = timeit.default_timer() - start_time
        backward_times.append(backward_time)

        print(f"Inference Time: {inference_time:.6f} seconds")
        print(f"Backward Time: {backward_time:.6f} seconds")

    (
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        if torch.cuda.is_available()
        else None
    )
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    return inference_times, backward_times


if __name__ == "__main__":
    main()
