use std::marker::PhantomData;

pub use graphpack_macros::graphpack;

pub struct Input<T> {
    _marker: PhantomData<T>,
}

/// Runs a serialized GraphDef in a local TensorFlow session and fetches an f32 output.
pub fn run_graph(
    graph_def: &[u8],
    inputs: &[(&str, &[f32])],
    output: &str,
) -> tensorflow::Result<tensorflow::Tensor<f32>> {
    run_graph_typed(graph_def, inputs, output)
}

/// Runs a serialized GraphDef in a local TensorFlow session and fetches an i32 output.
pub fn run_graph_i32(
    graph_def: &[u8],
    inputs: &[(&str, &[i32])],
    output: &str,
) -> tensorflow::Result<tensorflow::Tensor<i32>> {
    run_graph_typed(graph_def, inputs, output)
}

fn run_graph_typed<T: tensorflow::TensorType>(
    graph_def: &[u8],
    inputs: &[(&str, &[T])],
    output: &str,
) -> tensorflow::Result<tensorflow::Tensor<T>> {
    let mut graph = tensorflow::Graph::new();
    let options = tensorflow::ImportGraphDefOptions::new();
    graph.import_graph_def(graph_def, &options)?;

    let session_options = tensorflow::SessionOptions::new();
    let session = tensorflow::Session::new(&session_options, &graph)?;

    let tensors: Vec<tensorflow::Tensor<T>> = inputs
        .iter()
        .map(|(_, values)| {
            tensorflow::Tensor::new(&[values.len() as u64])
                .with_values(values)
                .expect("input values must match tensor shape")
        })
        .collect();

    let mut args = tensorflow::SessionRunArgs::new();
    for ((name, _), tensor) in inputs.iter().zip(tensors.iter()) {
        let operation = graph.operation_by_name_required(name)?;
        args.add_feed(&operation, 0, tensor);
    }

    let output_operation = graph.operation_by_name_required(output)?;
    let output_token = args.request_fetch(&output_operation, 0);
    session.run(&mut args)?;
    args.fetch(output_token)
}

#[cfg(test)]
mod tests {
    use super::{graphpack, run_graph, run_graph_i32, Input};

    #[test]
    fn graphpack_multi_line_arithmetic_runs() {
        let graph_def = graphpack!(|x: Input<f32>| {
            let a = x + 1.0;
            let b = a * 2.0;
            b - 3.0
        });

        let output = run_graph(&graph_def, &[("x", &[1.0, 2.0, 3.0])], "output").unwrap();
        assert_eq!(output.as_ref(), &[-1.0, 3.0, 7.0]);
    }

    #[test]
    fn graphpack_bitwise_runs() {
        let graph_def = graphpack!(|x: Input<i32>| {
            let a = x & 0xff;
            let b = a << 1;
            b | 1
        });

        let output = run_graph_i32(&graph_def, &[("x", &[1, 2, 3])], "output").unwrap();
        assert_eq!(output.as_ref(), &[3, 5, 7]);
    }

    #[test]
    fn graphpack_unary_arithmetic_runs() {
        let graph_def = graphpack!(|x: Input<f32>| {
            let y = -x;
            y + 2.0
        });

        let output = run_graph(&graph_def, &[("x", &[1.0, 2.0, 3.0])], "output").unwrap();
        assert_eq!(output.as_ref(), &[1.0, 0.0, -1.0]);
    }
}
