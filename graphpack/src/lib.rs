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
    let mut graph = tensorflow::Graph::new();
    let options = tensorflow::ImportGraphDefOptions::new();
    graph.import_graph_def(graph_def, &options)?;

    let session_options = tensorflow::SessionOptions::new();
    let session = tensorflow::Session::new(&session_options, &graph)?;

    let tensors: Vec<tensorflow::Tensor<f32>> = inputs
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
    use super::{graphpack, run_graph, Input};

    #[test]
    fn graphpack_input_graph_runs() {
        let graph_def = graphpack!(|x: Input<f32>| x + 1.0);

        let output = run_graph(&graph_def, &[("x", &[1.0, 2.0, 3.0])], "output").unwrap();

        assert_eq!(output.as_ref(), &[2.0, 3.0, 4.0]);
    }
}
