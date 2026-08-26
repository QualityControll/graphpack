mod graph;
mod graph_value;
mod input;
mod op;
mod tensorflow;

pub use graph::Graph;
pub use graph_value::GraphValue;
pub use input::{GraphSeq, Input, InputTupleMap};
pub use op::{ConstantValue, GraphType, Op, OpKind, ReduceKind, ScalarType};

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
        inputs: Vec<(&str, Tensor<T>)>,
    ) -> Tensor<U> {
        let output_op: Operation = graph.operation_by_name("output").unwrap().unwrap();
        let mut args = SessionRunArgs::new();
        for (name, input) in &inputs {
            let op = graph.operation_by_name(name).unwrap().unwrap();
            args.add_feed(&op, 0, input);
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
        let graph = x
            .map(|v| v * 2.0)
            .map(|v| v + 1.0)
            .map(|v| v * 3.0)
            .collect();
        let output: Tensor<f32> = run_graph(&graph, vec![("x", Tensor::from(3.0_f32))]);
        assert_eq!(output[0], 21.0);
    }
    #[test]
    fn multiple_inputs_can_be_mapped_together() {
        let x = Input::<i32>::new("x");
        let y = Input::<i32>::new("y");
        let graph = (x, y).map(|(x, y)| x + y).collect();
        let output: Tensor<i32> = run_graph(
            &graph,
            vec![("x", Tensor::from(3_i32)), ("y", Tensor::from(4_i32))],
        );
        assert_eq!(output[0], 7);
    }
    #[test]
    fn nested_tuples_can_be_destructured() {
        let a = Input::<i32>::new("a");
        let b = Input::<i32>::new("b");
        let c = Input::<i32>::new("c");
        let d = Input::<i32>::new("d");
        let graph = ((a, b), (c, d))
            .map(|((a, b), (c, d))| a * b + c * d)
            .collect();
        let output: Tensor<i32> = run_graph(
            &graph,
            vec![
                ("a", Tensor::from(2_i32)),
                ("b", Tensor::from(3_i32)),
                ("c", Tensor::from(4_i32)),
                ("d", Tensor::from(5_i32)),
            ],
        );
        assert_eq!(output[0], 26);
    }
    #[test]
    fn enumerate_returns_index_and_value_sequences() {
        let x = Input::<i32>::new("x");
        let (indices, values) = x.sequence().enumerate();
        let graph = indices.map(|i| i + 10).collect();
        let output: Tensor<i64> = run_graph(
            &graph,
            vec![("x", Tensor::new(&[4]).with_values(&[7, 8, 9, 10]).unwrap())],
        );
        assert_eq!(output.to_vec(), vec![10, 11, 12, 13]);
        let graph = values.map(|v| v + 1).collect();
        let output: Tensor<i32> = run_graph(
            &graph,
            vec![("x", Tensor::new(&[4]).with_values(&[7, 8, 9, 10]).unwrap())],
        );
        assert_eq!(output.to_vec(), vec![8, 9, 10, 11]);
    }
    #[test]
    fn zip_returns_left_and_right_sequences() {
        let x = Input::<i32>::new("x");
        let y = Input::<i32>::new("y");
        let (left, right) = x.sequence().zip(y.sequence());
        let graph = left.map(|v| v + 1).collect();
        let output: Tensor<i32> = run_graph(
            &graph,
            vec![
                ("x", Tensor::new(&[3]).with_values(&[1, 2, 3]).unwrap()),
                ("y", Tensor::new(&[3]).with_values(&[10, 20, 30]).unwrap()),
            ],
        );
        assert_eq!(output.to_vec(), vec![2, 3, 4]);
        let graph = right.map(|v| v + 1).collect();
        let output: Tensor<i32> = run_graph(
            &graph,
            vec![
                ("x", Tensor::new(&[3]).with_values(&[1, 2, 3]).unwrap()),
                ("y", Tensor::new(&[3]).with_values(&[10, 20, 30]).unwrap()),
            ],
        );
        assert_eq!(output.to_vec(), vec![11, 21, 31]);
    }
    #[test]
    fn filter_works() {
        let x = Input::<i32>::new("x");
        let graph = x.filter(|v| v.gt_scalar(2)).collect();
        let output: Tensor<i32> = run_graph(
            &graph,
            vec![(
                "x",
                Tensor::new(&[5]).with_values(&[1, 2, 3, 4, 5]).unwrap(),
            )],
        );
        assert_eq!(output.to_vec(), vec![3, 4, 5]);
    }
    #[test]
    fn take_works() {
        let x = Input::<i32>::new("x");
        let graph = x.take(3).collect();
        let output: Tensor<i32> = run_graph(
            &graph,
            vec![(
                "x",
                Tensor::new(&[5]).with_values(&[1, 2, 3, 4, 5]).unwrap(),
            )],
        );
        assert_eq!(output.to_vec(), vec![1, 2, 3]);
    }
    #[test]
    fn skip_works() {
        let x = Input::<i32>::new("x");
        let graph = x.skip(2).collect();
        let output: Tensor<i32> = run_graph(
            &graph,
            vec![(
                "x",
                Tensor::new(&[5]).with_values(&[1, 2, 3, 4, 5]).unwrap(),
            )],
        );
        assert_eq!(output.to_vec(), vec![3, 4, 5]);
    }
    #[test]
    fn reductions_work() {
        let x = Input::<i32>::new("x");
        let graph = (x.sequence().sum()).collect();
        let output: Tensor<i32> = run_graph(
            &graph,
            vec![("x", Tensor::new(&[4]).with_values(&[1, 2, 3, 4]).unwrap())],
        );
        assert_eq!(output[0], 10);
    }
    #[test]
    fn count_works() {
        let x = Input::<i32>::new("x");
        let graph = x.count().collect();
        let output: Tensor<i64> = run_graph(
            &graph,
            vec![("x", Tensor::new(&[4]).with_values(&[1, 2, 3, 4]).unwrap())],
        );
        assert_eq!(output[0], 4);
    }
    #[test]
    fn any_all_work() {
        let x = Input::<bool>::new("x");
        let any = x.sequence().any().collect();
        let all = x.sequence().all().collect();
        let out_any: Tensor<bool> = run_graph(
            &any,
            vec![(
                "x",
                Tensor::new(&[3])
                    .with_values(&[false, true, false])
                    .unwrap(),
            )],
        );
        let out_all: Tensor<bool> = run_graph(
            &all,
            vec![(
                "x",
                Tensor::new(&[3])
                    .with_values(&[false, true, false])
                    .unwrap(),
            )],
        );
        assert!(out_any[0]);
        assert!(!out_all[0]);
    }
    #[test]
    fn scalar_types_still_work() {
        let x = Input::<i8>::new("x");
        let graph = x.map(|v| v * 2_i8 + 1_i8).collect();
        let output: Tensor<i8> = run_graph(&graph, vec![("x", Tensor::from(3_i8))]);
        assert_eq!(output[0], 7);
    }
    #[allow(dead_code)]
    fn _keep_complex_type(_: Complex<f32>) {}
}
