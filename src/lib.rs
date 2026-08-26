mod graph;
mod graph_value;
mod input;
mod op;
mod scalars;
mod tensorflow;

pub use graph::Graph;
pub use graph_value::GraphValue;
pub use input::Input;
pub use op::{ConstantValue, GraphType, Op, OpKind, ScalarType};
pub use scalars::{Complex128, Complex64};

#[cfg(test)]
mod tests {
    use super::*;
    use ::tensorflow::{Graph as TensorFlowGraph, Operation, Session, SessionOptions, SessionRunArgs, Tensor, TensorType};
    use num_complex::Complex;

    fn run_graph<T: TensorType, U: TensorType>(
        graph: &TensorFlowGraph,
        input_name: &str,
        input: Tensor<T>,
    ) -> Tensor<U> {
        let x_op: Operation = graph.operation_by_name(input_name).unwrap().unwrap();
        let output_op: Operation = graph.operation_by_name("output").unwrap().unwrap();
        let mut args = SessionRunArgs::new();
        args.add_feed(&x_op, 0, &input);
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
    fn comparisons_work_in_map() {
        let x = Input::<i32>::new("x");
        let graph = x.map(|v| v.gt_scalar(10)).collect();
        let output: Tensor<bool> = run_graph(&graph, "x", Tensor::from(12_i32));
        assert!(output[0]);
    }

    #[test]
    fn all_comparisons_work_in_map() {
        let x = Input::<i32>::new("x");
        let graph = x.map(|v| v.ge_scalar(10)).collect();
        let output: Tensor<bool> = run_graph(&graph, "x", Tensor::from(10_i32));
        assert!(output[0]);
    }

    #[test]
    fn bitwise_operators_work_in_map() {
        let x = Input::<i32>::new("x");
        let graph = x.map(|v| ((v & 0b1111) ^ 0b0011) | 0b1000).collect();
        let output: Tensor<i32> = run_graph(&graph, "x", Tensor::from(0b0101_i32));
        assert_eq!(output[0], 0b1110);
    }

    #[test]
    fn bitwise_not_works_in_map() {
        let x = Input::<i32>::new("x");
        let graph = x.map(|v| !v).collect();
        let output: Tensor<i32> = run_graph(&graph, "x", Tensor::from(0_i32));
        assert_eq!(output[0], !0_i32);
    }

    #[test]
    fn bitwise_shifts_work_in_map() {
        let x = Input::<i32>::new("x");
        let graph = x.map(|v| (v << 2) >> 1).collect();
        let output: Tensor<i32> = run_graph(&graph, "x", Tensor::from(3_i32));
        assert_eq!(output[0], 6);
    }

    #[test]
    fn filter_works_with_map() {
        let x = Input::<i32>::new("x");
        let graph = x.filter(|v| v.gt_scalar(10)).map(|v| v * 2).collect();
        let mut input = Tensor::new(&[4]);
        input[0] = 5;
        input[1] = 12;
        input[2] = 20;
        input[3] = 3;
        let output: Tensor<i32> = run_graph(&graph, "x", input);
        assert_eq!(output.to_vec(), vec![24, 40]);
    }

    #[test]
    fn boolean_predicates_can_be_composed() {
        let x = Input::<i32>::new("x");
        let graph = x
            .filter(|v| v.gt_scalar(10).and(v.lt_scalar(20)))
            .collect();
        let mut input = Tensor::new(&[4]);
        input[0] = 5;
        input[1] = 12;
        input[2] = 20;
        input[3] = 15;
        let output: Tensor<i32> = run_graph(&graph, "x", input);
        assert_eq!(output.to_vec(), vec![12, 15]);
    }

    #[test]
    fn boolean_predicates_support_or_and_not() {
        let x = Input::<i32>::new("x");
        let graph = x
            .filter(|v| v.lt_scalar(6).or(v.gt_scalar(19)).not())
            .collect();
        let mut input = Tensor::new(&[4]);
        input[0] = 5;
        input[1] = 12;
        input[2] = 20;
        input[3] = 15;
        let output: Tensor<i32> = run_graph(&graph, "x", input);
        assert_eq!(output.to_vec(), vec![12, 15]);
    }

    #[test]
    fn complex64_works_in_map() {
        let x = Input::<Complex64>::new("x");
        let graph = x
            .map(|v| v * Complex::new(2.0_f32, 1.0_f32))
            .collect();
        let output: Tensor<Complex<f32>> = run_graph(
            &graph,
            "x",
            Tensor::from(Complex::new(1.0_f32, 2.0_f32)),
        );
        assert_eq!(output[0], Complex::new(0.0_f32, 5.0_f32));
    }

    #[test]
    fn complex128_works_in_map() {
        let x = Input::<Complex128>::new("x");
        let graph = x
            .map(|v| v * Complex::new(2.0_f64, 1.0_f64))
            .collect();
        let output: Tensor<Complex<f64>> = run_graph(
            &graph,
            "x",
            Tensor::from(Complex::new(1.0_f64, 0.0_f64)),
        );
        assert_eq!(output[0], Complex::new(2.0_f64, 1.0_f64));
    }

    #[test]
    fn complex64_constants_work_in_map() {
        let x = Input::<Complex64>::new("x");
        let graph = x
            .map(|v| v + Complex::new(1.0_f32, 2.0_f32))
            .collect();
        let output: Tensor<Complex<f32>> = run_graph(
            &graph,
            "x",
            Tensor::from(Complex::new(3.0_f32, 4.0_f32)),
        );
        assert_eq!(output[0], Complex::new(4.0_f32, 6.0_f32));
    }

    #[test]
    fn complex128_constants_work_in_map() {
        let x = Input::<Complex128>::new("x");
        let graph = x
            .map(|v| v + Complex::new(1.0_f64, 2.0_f64))
            .collect();
        let output: Tensor<Complex<f64>> = run_graph(
            &graph,
            "x",
            Tensor::from(Complex::new(3.0_f64, 4.0_f64)),
        );
        assert_eq!(output[0], Complex::new(4.0_f64, 6.0_f64));
    }

    #[test]
    fn string_comparison_works_in_map() {
        let x = Input::<String>::new("x");
        let graph = x.map(|v| v.eq_scalar("hello")).collect();
        let output: Tensor<bool> = run_graph(&graph, "x", Tensor::from("hello".to_string()));
        assert!(output[0]);
    }

    #[test]
    fn string_inequality_works_in_map() {
        let x = Input::<String>::new("x");
        let graph = x.map(|v| v.ne_scalar("hello")).collect();
        let output: Tensor<bool> = run_graph(&graph, "x", Tensor::from("world".to_string()));
        assert!(output[0]);
    }
}
