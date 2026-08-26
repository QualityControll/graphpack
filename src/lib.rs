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
    use tensorflow::{Session, SessionOptions, SessionRunArgs, Tensor};

    #[test]
    fn map_lowers_to_executable_tensorflow_graph() {
        let x = Input::<f32>::new("x");
        let graph = x.map(|v| v * 2.0 + 1.0).collect();

        assert!(graph.operation_by_name("x").unwrap().is_some());
        assert!(graph.operation_by_name("output").unwrap().is_some());

        let x_op = graph.operation_by_name("x").unwrap().unwrap();
        let output_op = graph.operation_by_name("output").unwrap().unwrap();

        let mut args = SessionRunArgs::new();
        args.add_feed(&x_op, 0, &Tensor::from(3.0_f32));
        let token = args.request_fetch(&output_op, 0);

        let session = Session::new(&SessionOptions::new(), &graph).unwrap();
        session.run(&mut args).unwrap();
        let output: Tensor<f32> = args.fetch(token).unwrap();

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

        assert!(graph.operation_by_name("x").unwrap().is_some());
        assert!(graph.operation_by_name("output").unwrap().is_some());
        assert_eq!(graph.operation_iter().count(), 7);
    }
}
