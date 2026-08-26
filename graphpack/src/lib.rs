use std::marker::PhantomData;

pub use graphpack_macros::graphpack;

pub struct Input<T> {
    _marker: PhantomData<T>,
}

/// Runs a serialized GraphDef in a local TensorFlow session, feeding f32 inputs.
///
/// The graph is executed without requesting outputs. This is useful for validating
/// that the generated graph can be imported and executed by TensorFlow.
pub fn run_graph(graph_def: &[u8], inputs: &[(&str, &[f32])]) -> tensorflow::Result<()> {
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
        args.add_target(&operation);
    }

    session.run(&mut args)
}

#[cfg(test)]
mod tests {
    use super::{graphpack, run_graph, Input};

    #[test]
    fn graphpack_input_graph_runs() {
        let graph_def = graphpack!(|x: Input<f32>| {});

        run_graph(&graph_def, &[("x", &[1.0, 2.0, 3.0])]).unwrap();
    }
}
