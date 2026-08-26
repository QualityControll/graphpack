mod graph_value;
mod input;
mod op;

pub use graph_value::GraphValue;
pub use input::Input;
pub use op::{Op, OpKind};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_closure_can_build_graph_with_regular_constants() {
        let x = Input::<f32>::new("x");
        let y = x.map(|v| v * 2.0 + 1.0);

        assert_eq!(y.op().kind(), &OpKind::Map);
    }
}
