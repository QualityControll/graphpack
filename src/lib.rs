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
    fn map_closure_can_be_materialized() {
        let x = Input::<f32>::new("x");
        let y = x.map(|v| v * 2.0 + 1.0);
        let graph = y.materialize();

        assert_eq!(graph.output().kind(), &OpKind::Map);
        assert_eq!(graph.operations().len(), 6);
    }
}
