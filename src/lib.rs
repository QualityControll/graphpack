mod graph;
mod graph_value;
mod input;
mod op;
mod tensorflow;

pub use graph::Graph;
pub use graph_value::{GraphValue, GraphValueTupleMap};
pub use input::{Input, InputTupleMap};
pub use op::{ConstantValue, GraphType, Op, OpKind, ScalarType};

#[cfg(test)]
mod tests {
    use super::*;
    use ::tensorflow::{Graph as TensorFlowGraph, Operation, Session, SessionOptions, SessionRunArgs, Tensor, TensorType};
    use num_complex::Complex;

    fn run_graph<T: TensorType, U: TensorType>(graph: &TensorFlowGraph, inputs: Vec<(&str, Tensor<T>)>) -> Tensor<U> {
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
        let output: Tensor<f32> = run_graph(&graph, vec![("x", Tensor::from(3.0_f32))]);
        assert_eq!(output[0], 7.0);
    }

    #[test]
    fn chained_map_lowers_to_tensorflow_graph() {
        let x = Input::<f32>::new("x");
        let graph = x.map(|v| v * 2.0).map(|v| v + 1.0).map(|v| v * 3.0).collect();
        let output: Tensor<f32> = run_graph(&graph, vec![("x", Tensor::from(3.0_f32))]);
        assert_eq!(output[0], 21.0);
    }

    #[test]
    fn multiple_inputs_can_be_mapped_together() {
        let x = Input::<i32>::new("x");
        let y = Input::<i32>::new("y");
        let graph = (x, y).map(|(x, y)| x + y).collect();
        let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::from(3_i32)), ("y", Tensor::from(4_i32))]);
        assert_eq!(output[0], 7);
    }

    #[test]
    fn multiple_inputs_can_be_used_in_a_complex_expression() {
        let x = Input::<i32>::new("x");
        let y = Input::<i32>::new("y");
        let graph = (x, y).map(|(x, y)| x * 2 + y * 3).collect();
        let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::from(3_i32)), ("y", Tensor::from(4_i32))]);
        assert_eq!(output[0], 18);
    }

    #[test]
    fn multiple_inputs_support_eight_inputs() {
        let inputs = (
            Input::<i32>::new("a"), Input::<i32>::new("b"), Input::<i32>::new("c"), Input::<i32>::new("d"),
            Input::<i32>::new("e"), Input::<i32>::new("f"), Input::<i32>::new("g"), Input::<i32>::new("h"),
        );
        let graph = inputs.map(|(a, b, c, d, e, f, g, h)| a + b + c + d + e + f + g + h).collect();
        let output: Tensor<i32> = run_graph(&graph, vec![
            ("a", Tensor::from(1_i32)), ("b", Tensor::from(2_i32)), ("c", Tensor::from(3_i32)), ("d", Tensor::from(4_i32)),
            ("e", Tensor::from(5_i32)), ("f", Tensor::from(6_i32)), ("g", Tensor::from(7_i32)), ("h", Tensor::from(8_i32)),
        ]);
        assert_eq!(output[0], 36);
    }

    #[test]
    fn nested_tuples_can_be_destructured() {
        let a = Input::<i32>::new("a"); let b = Input::<i32>::new("b");
        let c = Input::<i32>::new("c"); let d = Input::<i32>::new("d");
        let graph = ((a, b), (c, d)).map(|((a, b), (c, d))| a * b + c * d).collect();
        let output: Tensor<i32> = run_graph(&graph, vec![("a", Tensor::from(2_i32)), ("b", Tensor::from(3_i32)), ("c", Tensor::from(4_i32)), ("d", Tensor::from(5_i32))]);
        assert_eq!(output[0], 26);
    }

    #[test]
    fn nested_four_input_tuples_can_be_destructured() {
        let a = Input::<i32>::new("a"); let b = Input::<i32>::new("b"); let c = Input::<i32>::new("c"); let d = Input::<i32>::new("d");
        let e = Input::<i32>::new("e"); let f = Input::<i32>::new("f"); let g = Input::<i32>::new("g"); let h = Input::<i32>::new("h");
        let graph = ((a, b, c, d), (e, f, g, h)).map(|((a, b, c, d), (e, f, g, h))| a + b * c + d + e + f * g + h).collect();
        let output: Tensor<i32> = run_graph(&graph, vec![
            ("a", Tensor::from(1_i32)), ("b", Tensor::from(2_i32)), ("c", Tensor::from(3_i32)), ("d", Tensor::from(4_i32)),
            ("e", Tensor::from(5_i32)), ("f", Tensor::from(6_i32)), ("g", Tensor::from(7_i32)), ("h", Tensor::from(8_i32)),
        ]);
        assert_eq!(output[0], 66);
    }

    #[test]
    fn tuple_inputs_can_be_reused_in_richer_expressions() {
        let x = Input::<i32>::new("x");
        let y = Input::<i32>::new("y");
        let graph = (x, y).map(|(x, y)| (x * x) + (y * y) + (x * y)).collect();
        let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::from(2_i32)), ("y", Tensor::from(3_i32))]);
        assert_eq!(output[0], 19);
    }

    #[test]
    fn tuple_valued_map_results_can_be_destructured() {
        let x = Input::<i32>::new("x");
        let tuple = x.map(|v| (v + 1, v * 2));
        let graph = tuple.map(|(a, b)| a * b).collect();
        let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::from(4_i32))]);
        assert_eq!(output[0], 40);
    }

    #[test]
    fn tuple_valued_map_results_can_be_chained() {
        let x = Input::<i32>::new("x");
        let tuple = x.map(|v| (v + 1, v * 2));
        let graph = tuple.map(|(a, b)| (a + b, a * b)).map(|(sum, product)| sum + product).collect();
        let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::from(4_i32))]);
        assert_eq!(output[0], 50);
    }

    #[test]
    fn tuple_valued_map_results_support_three_values() {
        let x = Input::<i32>::new("x");
        let tuple = x.map(|v| (v, v + 1, v + 2));
        let graph = tuple.map(|(a, b, c)| a + b + c).collect();
        let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::from(4_i32))]);
        assert_eq!(output[0], 15);
    }

    #[test]
    fn complex_scalar_types_work() {
        let x = Input::<Complex<f32>>::new("x");
        let graph = x.map(|v| v + Complex::new(1.0, 2.0)).collect();
        let output: Tensor<Complex<f32>> = run_graph(&graph, vec![("x", Tensor::from(Complex::new(3.0_f32, 4.0_f32)))]);
        assert_eq!(output[0], Complex::new(4.0, 6.0));
    }

    #[test]
    fn string_scalar_type_works() {
        let x = Input::<String>::new("x");
        let graph = x.map(|v| v.eq_scalar("hello")).collect();
        let input = Tensor::new(&[1]).with_values(&["hello".to_string()]).unwrap();
        let output: Tensor<bool> = run_graph(&graph, vec![("x", input)]);
        assert!(output[0]);
    }
}
