mod graph;
mod graph_value;
mod input;
mod op;

pub use graph::Graph;
pub use graph_value::GraphValue;
pub use input::Input;
pub use op::{ConstantValue, Op, OpKind};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_closure_builds_graph_with_regular_constants() {
        let x = Input::<f32>::new("x");
        let y = x.map(|v| v * 2.0 + 1.0);
        let graph = y.collect();

        assert_eq!(graph.output().kind(), &OpKind::Add);
        assert_eq!(graph.operations().len(), 5);
    }

    #[test]
    fn map_can_be_chained() {
        let x = Input::<f32>::new("x");
        let y = x
            .map(|v| v * 2.0)
            .map(|v| v + 1.0)
            .map(|v| v * 3.0);
        let graph = y.collect();

        assert_eq!(graph.output().kind(), &OpKind::Mul);
        assert_eq!(graph.operations().len(), 7);
    }
}
