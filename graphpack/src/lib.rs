use std::marker::PhantomData;

pub use graphpack_macros::graphpack;

pub struct Input<T> {
    _marker: PhantomData<T>,
}

/// Runs a serialized GraphDef in a local TensorFlow session and fetches an f32 output.
fn run_graph<T: tensorflow::TensorType>(
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
    use std::marker::PhantomData;

    use super::{graphpack, run_graph, Input};

    // The macro only needs the Rust type syntax to identify Complex<f32> when
    // constructing the graph. TensorFlow's runtime dtype is validated below.
    struct Complex<T>(PhantomData<T>);

    #[test]
    fn graphpack_multi_line_arithmetic_runs() {
        let graph_def = graphpack!(|x: Input<f32>| {
            let a = x + 1.0;
            let b = a * 2.0;
            b - 3.0
        });

        let output = run_graph(&graph_def, &[("x", &[1.0f32, 2.0, 3.0])], "output").unwrap();
        assert_eq!(output.as_ref(), &[1.0f32, 3.0, 5.0]);
    }

    #[test]
    fn graphpack_multiple_f32_inputs_runs() {
        let graph_def = graphpack!(|x: Input<f32>, y: Input<f32>| {
            let sum = x + y;
            sum * 2.0
        });

        let output = run_graph(
            &graph_def,
            &[("x", &[1.0f32, 2.0, 3.0]), ("y", &[4.0, 5.0, 6.0])],
            "output",
        )
        .unwrap();
        assert_eq!(output.as_ref(), &[10.0f32, 14.0, 18.0]);
    }

    #[test]
    fn graphpack_multiple_i32_inputs_runs() {
        let graph_def = graphpack!(|x: Input<i32>, y: Input<i32>| {
            let difference = x - y;
            difference * 2
        });

        let output = run_graph(
            &graph_def,
            &[("x", &[10, 20, 30]), ("y", &[1, 2, 3])],
            "output",
        )
        .unwrap();
        assert_eq!(output.as_ref(), &[18, 36, 54]);
    }

    #[test]
    fn graphpack_bitwise_runs() {
        let graph_def = graphpack!(|x: Input<i32>| {
            let a = x & 0xff;
            let b = a << 1;
            b | 1
        });

        let output = run_graph(&graph_def, &[("x", &[1, 2, 3])], "output").unwrap();
        assert_eq!(output.as_ref(), &[3, 5, 7]);
    }

    #[test]
    fn graphpack_unary_arithmetic_runs() {
        let graph_def = graphpack!(|x: Input<f32>| {
            let y = -x;
            y + 2.0
        });

        let output = run_graph(&graph_def, &[("x", &[1.0f32, 2.0, 3.0])], "output").unwrap();
        assert_eq!(output.as_ref(), &[1.0f32, 0.0, -1.0]);
    }

    #[test]
    fn graphpack_if_else_runs() {
        let graph_def = graphpack!(|x: Input<f32>| {
            if x > 0.0 { x * 2.0 } else { x - 2.0 }
        });

        let output = run_graph(&graph_def, &[("x", &[1.0f32, -2.0, 3.0])], "output").unwrap();
        assert_eq!(output.as_ref(), &[2.0f32, -4.0, 6.0]);
    }

    #[test]
    fn graphpack_if_else_with_let_bindings_runs() {
        let graph_def = graphpack!(|x: Input<f32>| {
            let doubled = x * 2.0;
            if x > 0.0 { doubled + 1.0 } else { doubled - 1.0 }
        });

        let output = run_graph(&graph_def, &[("x", &[1.0f32, -2.0])], "output").unwrap();
        assert_eq!(output.as_ref(), &[3.0f32, -5.0]);
    }

    #[test]
    fn graphpack_complex64_input_has_complex_dtype() {
        let graph_def = graphpack!(|x: Input<Complex<f32>>| x + x);

        let mut graph = tensorflow::Graph::new();
        let options = tensorflow::ImportGraphDefOptions::new();
        graph.import_graph_def(&graph_def, &options).unwrap();

        let input = graph.operation_by_name_required("x").unwrap();
        let output = graph.operation_by_name_required("output").unwrap();

        assert_eq!(input.output_type(0), tensorflow::DataType::Complex64);
        assert_eq!(output.output_type(0), tensorflow::DataType::Complex64);
    }

    #[test]
    fn graphpack_multiple_complex64_inputs_have_complex_dtype() {
        let graph_def = graphpack!(|x: Input<Complex<f32>>, y: Input<Complex<f32>>| {
            let sum = x + y;
            sum * x
        });

        let mut graph = tensorflow::Graph::new();
        let options = tensorflow::ImportGraphDefOptions::new();
        graph.import_graph_def(&graph_def, &options).unwrap();

        assert_eq!(
            graph.operation_by_name_required("x").unwrap().output_type(0),
            tensorflow::DataType::Complex64
        );
        assert_eq!(
            graph.operation_by_name_required("y").unwrap().output_type(0),
            tensorflow::DataType::Complex64
        );
        assert_eq!(
            graph
                .operation_by_name_required("output")
                .unwrap()
                .output_type(0),
            tensorflow::DataType::Complex64
        );
    }
}
