mod graph;
mod graph_value;
mod input;
mod op;
mod tensorflow;

pub use graph::Graph;
pub use graph_value::GraphValue;
pub use input::Input;
pub use op::{ConstantValue, GraphType, Op, OpKind, ScalarType};

#[cfg(test)]
mod tests {
    use super::*;
    use ::tensorflow::{Graph as TensorFlowGraph, Operation, Session, SessionOptions, SessionRunArgs, Tensor, TensorType};

    fn run_graph<T: TensorType>(graph: &TensorFlowGraph, input_name: &str, input: Tensor<T>) -> Tensor<T> {
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

        assert!(graph.operation_by_name("x").is_ok());
        assert!(graph.operation_by_name("output").is_ok());

        let output = run_graph(&graph, "x", Tensor::from(3.0_f32));
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

        assert!(graph.operation_by_name("x").is_ok());
        assert!(graph.operation_by_name("output").is_ok());
        assert_eq!(graph.operation_iter().count(), 7);

        let output = run_graph(&graph, "x", Tensor::from(3.0_f32));
        assert_eq!(output[0], 21.0);
    }

    #[test]
    fn bitwise_operators_work_in_map() {
        let x = Input::<i32>::new("x");
        let graph = x
            .map(|v| ((v & 0b1111) ^ 0b0011) | 0b1000)
            .collect();

        let output = run_graph(&graph, "x", Tensor::from(0b0101_i32));
        assert_eq!(output[0], 0b1110);
    }

    #[test]
    fn bitwise_not_works_in_map() {
        let x = Input::<i32>::new("x");
        let graph = x.map(|v| !v).collect();

        let output = run_graph(&graph, "x", Tensor::from(0_i32));
        assert_eq!(output[0], !0_i32);
    }

    #[test]
    fn bitwise_shifts_work_in_map() {
        let x = Input::<i32>::new("x");
        let graph = x.map(|v| (v << 2) >> 1).collect();

        let output = run_graph(&graph, "x", Tensor::from(3_i32));
        assert_eq!(output[0], 6);
    }
}
