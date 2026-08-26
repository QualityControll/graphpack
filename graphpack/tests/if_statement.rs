use graphpack::{graphpack, Input};

#[test]
fn if_statement_without_value_is_supported() {
    let graph_def = graphpack!(|x: Input<f32>| {
        let doubled = x * 2.0;
        if x > 0.0 {
            doubled + 1.0;
        } else {
            doubled - 1.0;
        }
        doubled + 10.0
    });

    let mut graph = tensorflow::Graph::new();
    let options = tensorflow::ImportGraphDefOptions::new();
    graph.import_graph_def(&graph_def, &options).unwrap();
    assert!(graph.operation_by_name("output").is_some());
}

#[test]
fn if_statement_can_be_multiline() {
    let graph_def = graphpack!(|x: Input<f32>| {
        if x > 0.0 {
            x + 1.0;
        } else {
            x - 1.0;
        }
        x * 2.0
    });

    let mut graph = tensorflow::Graph::new();
    let options = tensorflow::ImportGraphDefOptions::new();
    graph.import_graph_def(&graph_def, &options).unwrap();
    assert!(graph.operation_by_name("output").is_some());
}
