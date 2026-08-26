use graphpack::{GraphValue, Input};
use tensorflow::{Graph as TensorFlowGraph, Operation, Session, SessionOptions, SessionRunArgs, Tensor, TensorType};

fn run_graph<T: TensorType, U: TensorType>(graph: &TensorFlowGraph, inputs: Vec<(&str, Tensor<T>)>) -> Tensor<U> {
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
fn sum_composes_after_filter_map() {
    let x = Input::<i32>::new("x");
    let graph = x.sequence().filter(|v| v.gt_scalar(1)).map(|v| v * 2).sum().collect();
    let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::new(&[4]).with_values(&[1, 2, 3, 4]).unwrap())]);
    assert_eq!(output[0], 18);
}

#[test]
fn product_composes_after_skip_take() {
    let x = Input::<i32>::new("x");
    let graph = x.sequence().skip(1).take(2).product().collect();
    let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::new(&[4]).with_values(&[2, 3, 4, 5]).unwrap())]);
    assert_eq!(output[0], 12);
}

#[test]
fn min_max_compose_after_enumerate_map() {
    let x = Input::<i64>::new("x");
    let graph = x.sequence().enumerate().map(|(i, v)| v + i).max().collect();
    let output: Tensor<i64> = run_graph(&graph, vec![("x", Tensor::new(&[3]).with_values(&[10, 5, 20]).unwrap())]);
    assert_eq!(output[0], 22);
}

#[test]
fn count_composes_after_zip_filter() {
    let x = Input::<i32>::new("x");
    let y = Input::<i32>::new("y");
    let graph = x.sequence().zip(y.sequence()).filter(|(a, _b)| a.gt_scalar(1)).map(|(a, b)| a + b).count().collect();
    let output: Tensor<i64> = run_graph(&graph, vec![
        ("x", Tensor::new(&[4]).with_values(&[1, 2, 3, 4]).unwrap()),
        ("y", Tensor::new(&[4]).with_values(&[10, 20, 30, 40]).unwrap()),
    ]);
    assert_eq!(output[0], 3);
}

#[test]
fn fold_composes_after_adapter_chain() {
    let x = Input::<i32>::new("x");
    let graph = x.sequence().skip(1).filter(|v| v.gt_scalar(1)).map(|v| v * 2).take(2).fold(10, |acc, v| acc + v).collect();
    let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::new(&[4]).with_values(&[1, 2, 3, 4]).unwrap())]);
    assert_eq!(output[0], 20);
}

#[test]
fn reduce_composes_after_arbitrary_adapter_chain() {
    let x = Input::<i64>::new("x");
    let graph = x.sequence().enumerate().filter(|(_i, v)| v.gt_scalar(1)).map(|(i, v)| v + i).reduce(|a, b| a + b).collect();
    let output: Tensor<i64> = run_graph(&graph, vec![("x", Tensor::new(&[3]).with_values(&[1_i64, 2, 3]).unwrap())]);
    assert_eq!(output[0], 8);
}

#[test]
fn boolean_reduction_composes_after_filter() {
    let x = Input::<bool>::new("x");
    let graph = x.sequence().filter(|v| v).reduce(|a, b| a.and(b)).collect();
    let output: Tensor<bool> = run_graph(&graph, vec![("x", Tensor::new(&[3]).with_values(&[true, true, false]).unwrap())]);
    assert!(!output[0]);
}
