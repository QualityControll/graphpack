use graphpack::GraphValue;
use graphpack::Input;

#[test]
fn map_closure_can_build_graph_with_regular_constants() {
    let x = Input::<f32>::new("x");

    let _y = x.map(|v| v * 2.0 + 1.0);
}
