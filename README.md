# GraphPack

GraphPack is a Rust-native computation graph library for building composable data-processing and numerical computations using ordinary Rust syntax. Instead of executing values immediately, GraphPack constructs a graph of operations over typed graph values and can lower that graph to TensorFlow for execution.

The goal is to make graph construction feel like programming against normal Rust values while retaining the ability to serialize, transport, and execute the resulting computation independently of the host application.

## What we're building

GraphPack is centered around a few ideas:

- **Rust-native graph construction** — write expressions and closures using `Input<T>`, `GraphValue<T>`, and sequence adapters.
- **Composable pipelines** — adapters such as `map`, `filter`, `take`, `skip`, `enumerate`, and `zip` can be chained together.
- **Terminal operations** — pipelines can terminate in reductions such as `fold`, `reduce`, `sum`, `product`, `min`, `max`, `any`, `all`, and `count`.
- **Typed graphs** — Rust scalar types are carried through GraphPack's type system and mapped to the appropriate TensorFlow dtypes.
- **TensorFlow execution** — GraphPack lowers the graph to TensorFlow operations and executes the resulting graph.
- **Portable computation** — the longer-term goal is to make a constructed graph an executable computation that can be serialized and sent to another process or machine.

The intended programming model is:

```text
Rust closures and graph values
              |
              v
        GraphPack graph
              |
              v
       TensorFlow lowering
              |
              v
          execution
```

## Example

A sequence can be transformed through several adapters and then terminated with a reduction:

```rust
use graphpack::{run_graph, Input};

fn main() {
    let x = Input::<i64>::new("x");

    let graph = x
        .sequence()
        .enumerate()
        .map(|(i, v)| v + i)
        .filter(|v| v.gt_scalar(10))
        .take(100)
        .sum()
        .collect();

    let result = run_graph(
        &graph,
        vec![("x", tensorflow::Tensor::<i64>::new(&[5])
            .with_values(&[10, 20, 5, 30, 15])
            .unwrap())],
    );

    println!("{result:?}");
}
```

The important part is the composition: the input is enumerated, mapped, filtered, limited, and finally reduced without manually constructing the underlying TensorFlow graph.

## Example goal: remote signal processing

One of the end goals for GraphPack is a remote signal-processing pipeline such as vibration analysis. A user should eventually be able to describe the computation in a natural Rust-style chain and execute it remotely with asynchronous execution:

```rust
let result = remote
    .input::<f32>("vibration")
    .window(WindowFunction::Hann)
    .fft()
    .map(|x| x.abs())
    .top_k(10)
    .execute()
    .await?;
```

This example is intentionally a goal for the future API. It illustrates the desired separation between describing a computation and executing it: `fft()` and `top_k()` are backend primitives, while `map()` remains a composable GraphPack operation, and `execute().await` submits the resulting computation to a remote executor.

## Roadmap

### Milestone 1 — Core graph DSL

- [x] Typed graph inputs and graph values
- [x] Scalar arithmetic and comparisons
- [x] Closure-based graph construction
- [x] Sequence adapters
- [x] Tuple and structured graph values
- [x] Basic control-flow and statement support

### Milestone 2 — Composable sequence operations

- [x] `map`
- [x] `filter`
- [x] `take`
- [x] `skip`
- [x] `enumerate`
- [x] `zip`
- [ ] Expand adapter coverage
- [ ] Ensure arbitrary compatible adapter chains compose cleanly

### Milestone 3 — Terminal and reduction operations

- [x] `sum`
- [x] `product`
- [x] `min`
- [x] `max`
- [x] `count`
- [x] `fold`
- [x] `reduce`
- [x] `any`
- [x] `all`
- [ ] Generalize reductions beyond the currently supported operation patterns
- [ ] Add comprehensive composition tests for every terminal operation

### Milestone 4 — Type system and correctness

- [x] `bool`
- [x] `i16`
- [x] `i32`
- [x] `i64`
- [x] `f32`
- [x] `f64`
- [x] `usize`
- [ ] Consistent type propagation through every adapter
- [ ] Better compile-time validation of incompatible operations
- [ ] Eliminate avoidable TensorFlow runtime dtype errors

### Milestone 5 — Numerical backend primitives

- [ ] First-class `Complex<f32>`
- [ ] Complex tensor/sequence support
- [ ] `fft`
- [ ] `ifft`
- [ ] `rfft`
- [ ] `irfft`
- [ ] Complex operations such as `conj`, `real`, `imag`, `abs`, and `angle`
- [ ] `reshape`
- [ ] `transpose`
- [ ] `slice`
- [ ] `gather`
- [ ] `concat`
- [ ] `stack`
- [ ] `matmul`
- [ ] `dot`
- [ ] `mean`
- [ ] `norm`
- [ ] `argmin` / `argmax`
- [ ] Tier 2 primitives such as `solve`, `inverse`, `det`, `svd`, `qr`, `eigh`, convolution, sorting, and `top_k`
- [ ] Complete an end-to-end `Complex<f32>` FFT example

### Milestone 6 — TensorFlow backend

- [x] Lower core graph operations to TensorFlow
- [x] Execute generated graphs locally
- [ ] Broaden TensorFlow operation coverage
- [ ] Improve lowering diagnostics
- [ ] Make graph lowering deterministic and robust

### Milestone 7 — Portable graphs

- [ ] Define a stable serialized GraphPack representation
- [ ] Serialize graph structure and type information
- [ ] Deserialize graphs independently of the original Rust process
- [ ] Validate serialized graphs before execution
- [ ] Version the serialized representation

### Milestone 8 — Remote execution

- [ ] Define a remote execution protocol
- [ ] Send serialized GraphPack graphs to a remote executor
- [ ] Bind remote inputs and retrieve results
- [ ] Support asynchronous `.await` execution
- [ ] Add a reference remote execution service
- [ ] Demonstrate execution on a separate machine
- [ ] Complete the remote vibration-analysis example

### Milestone 9 — GraphPack 1.0

- [ ] Stable public API
- [ ] Comprehensive unit and integration test suite
- [ ] Documentation for graph construction and execution
- [ ] Portable serialized computation format
- [ ] Local and remote execution paths
- [ ] Performance benchmarks
- [ ] Examples covering realistic numerical/data-processing workloads

## Design goal

The end goal is simple:

> **Write a computation in Rust once, compose it naturally, turn it into a portable computation graph, and execute that graph wherever the GraphPack runtime is available.**

GraphPack should make the graph itself the reusable artifact — not the TensorFlow implementation details used to execute it.
