mod graph;
mod graph_value;
mod input;
mod op;
mod tensorflow;

pub use graph::Graph;
pub use graph_value::GraphValue;
pub use input::{Input, InputTupleMap};
pub use op::{ConstantValue, GraphType, Op, OpKind, ScalarType};

#[cfg(test)]
mod tests {
    use super::*;
    use ::tensorflow::{
        Graph as TensorFlowGraph, Operation, Session, SessionOptions, SessionRunArgs, Tensor,
        TensorType,
    };
    use num_complex::Complex;

    fn run_graph<T: TensorType, U: TensorType>(
        graph: &TensorFlowGraph,
        input_name: &str,
        input: Tensor<T>,
    ) -> Tensor<U> {
        let x_op: Operation = graph.operation_by_name(input_name).unwrap().unwrap();
        let output_op: Operation = match graph.operation_by_name("output") {
            Ok(Some(operation)) => operation,
            Ok(None) => x_op.clone(),
            Err(error) => panic!("failed to find output operation: {error}"),
        };
        let mut args = SessionRunArgs::new();
        args.add_feed(&x_op, 0, &input);
        let token = args.request_fetch(&output_op, 0);
        let session = Session::new(&SessionOptions::new(), graph).unwrap();
        session.run(&mut args).unwrap();
        args.fetch(token).unwrap()
    }

    fn run_graph_with_inputs(
        graph: &TensorFlowGraph,
        inputs: Vec<(&str, Tensor<i32>)>,
    ) -> Tensor<i32> {
        let output_op: Operation = graph.operation_by_name("output").unwrap().unwrap();
        let mut input_ops = Vec::with_capacity(inputs.len());

        for (name, input) in inputs {
            let input_op = graph.operation_by_name(name).unwrap().unwrap();
            input_ops.push((input_op, input));
        }

        let mut args = SessionRunArgs::new();
        for (input_op, input) in &input_ops {
            args.add_feed(input_op, 0, input);
        }

        let token = args.request_fetch(&output_op, 0);
        let session = Session::new(&SessionOptions::new(), graph).unwrap();
        session.run(&mut args).unwrap();
        args.fetch(token).unwrap()
    }

    #[test]
    fn map_lowers_to_executable_tensorflow_graph() {
        let x = Input::<f32>::new("x");
        let graph = x.map(|v| v * 2.0 + 1.0).collect();
        let output: Tensor<f32> = run_graph(&graph, "x", Tensor::from(3.0_f32));
        assert_eq!(output[0], 7.0);
    }

    #[test]
    fn chained_map_lowers_to_tensorflow_graph() {
        let x = Input::<f32>::new("x");
        let graph = x
            .map(|v| v * 2.0)
            .map(|v| v + 1.0)
            .map(|v| v * 3.0)
            .collect();
        let output: Tensor<f32> = run_graph(&graph, "x", Tensor::from(3.0_f32));
        assert_eq!(output[0], 21.0);
    }

    #[test]
    fn multiple_inputs_can_be_mapped_together() {
        let x = Input::<i32>::new("x");
        let y = Input::<i32>::new("y");
        let graph = (x, y).map(|(x, y)| x + y).collect();
        let output = run_graph_with_inputs(
            &graph,
            vec![("x", Tensor::from(3_i32)), ("y", Tensor::from(4_i32))],
        );
        assert_eq!(output[0], 7);
    }

    #[test]
    fn multiple_inputs_can_be_used_in_a_complex_expression() {
        let x = Input::<i32>::new("x");
        let y = Input::<i32>::new("y");
        let graph = (x, y).map(|(x, y)| x * 2 + y * 3).collect();
        let output = run_graph_with_inputs(
            &graph,
            vec![("x", Tensor::from(3_i32)), ("y", Tensor::from(4_i32))],
        );
        assert_eq!(output[0], 18);
    }

    #[test]
    fn multiple_inputs_support_eight_inputs() {
        let inputs = (
            Input::<i32>::new("a"),
            Input::<i32>::new("b"),
            Input::<i32>::new("c"),
            Input::<i32>::new("d"),
            Input::<i32>::new("e"),
            Input::<i32>::new("f"),
            Input::<i32>::new("g"),
            Input::<i32>::new("h"),
        );
        let graph = inputs
            .map(|(a, b, c, d, e, f, g, h)| a + b + c + d + e + f + g + h)
            .collect();
        let output = run_graph_with_inputs(
            &graph,
            vec![
                ("a", Tensor::from(1_i32)),
                ("b", Tensor::from(2_i32)),
                ("c", Tensor::from(3_i32)),
                ("d", Tensor::from(4_i32)),
                ("e", Tensor::from(5_i32)),
                ("f", Tensor::from(6_i32)),
                ("g", Tensor::from(7_i32)),
                ("h", Tensor::from(8_i32)),
            ],
        );
        assert_eq!(output[0], 36);
    }

    #[test]
    fn complex_scalar_types_work() {
        let x = Input::<Complex<f32>>::new("x");
        let graph = x.map(|v| v + Complex::new(1.0, 2.0)).collect();
        let output: Tensor<Complex<f32>> = run_graph(
            &graph,
            "x",
            Tensor::from(Complex::new(3.0_f32, 4.0_f32)),
        );
        assert_eq!(output[0], Complex::new(4.0, 6.0));
    }

    #[test]
    fn string_scalar_type_works() {
        let x = Input::<String>::new("x");
        let graph = x.map(|v| v).collect();
        let input = Tensor::new(&[1]).with_values(&["hello".to_string()]).unwrap();
        let output: Tensor<String> = run_graph(&graph, "x", input);
        assert_eq!(output[0], "hello");
    }
}
