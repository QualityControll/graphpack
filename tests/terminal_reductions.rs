use graphpack::Input;
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
fn sum_composes_after_arbitrary_adapter_chain() {
    let x = Input::<i64>::new("x");
    let graph = x.sequence().skip(1).take(3).enumerate().filter(|(_i, v)| v.gt_scalar(15)).map(|(i, v)| v + i).sum().collect();
    let output: Tensor<i64> = run_graph(&graph, vec![("x", Tensor::new(&[5]).with_values(&[10_i64, 20, 30, 40, 50]).unwrap())]);
    assert_eq!(output[0], 93);
}

#[test]
fn product_composes_after_filter_and_map() {
    let x = Input::<i32>::new("x");
    let graph = x.sequence().filter(|v| v.gt_scalar(1)).map(|v| v + 1).product().collect();
    let output: Tensor<i32> = run_graph(&graph, vec![("x", Tensor::new(&[4]).with_values(&[1, 2, 3, 4]).unwrap())]);
    assert_eq!(output[0], 60);
}

#[test]
fn min_and_max_compose_after_zip_and_map() {
    let x = Input::<i32>::new("x");
    let y = Input::<i32>::new("y");
    let graph_min = x.sequence().zip(y.sequence()).map(|(a, b)| a * 10 + b).min().collect();
    let output_min: Tensor<i32> = run_graph(&graph_min, vec![("x", Tensor::new(&[3]).with_values(&[3, 1, 2]).unwrap()), ("y", Tensor::new(&[3]).with_values(&[4, 5, 6]).unwrap())]);
    assert_eq!(output_min[0], 16);
    let x = Input::<i32>::new("x");
    let y = Input::<i32>::new("y");
    let graph_max = x.sequence().zip(y.sequence()).map(|(a, b)| a * 10 + b).max().collect();
    let output_max: Tensor<i32> = run_graph(&graph_max, vec![("x", Tensor::new(&[3]).with_values(&[3, 1, 2]).unwrap()), ("y", Tensor::new(&[3]).with_values(&[4, 5, 6]).unwrap())]);
    assert_eq!(output_max[0], 34);
}

#[test]
fn count_composes_after_arbitrary_adapters() {
    let x = Input::<i32>::new("x");
    let y = Input::<i32>::new("y");
    let graph = x.sequence().zip(y.sequence()).skip(1).take(2).filter(|(a, _b)| a.gt_scalar(1)).map(|(a, b)| a + b).count().collect();
    let output: Tensor<i64> = run_graph(&graph, vec![("x", Tensor::new(&[4]).with_values(&[1, 2, 3, 4]).unwrap()), ("y", Tensor::new(&[4]).with_values(&[10, 20, 30, 40]).unwrap())]);
    assert_eq!(output[0], 2);
}

#[test]
fn any_and_all_compose_after_map_and_filter() {
    let x = Input::<i32>::new("x");
    let graph_any = x.sequence().filter(|v| v.gt_scalar(1)).map(|v| v.gt_scalar(3)).any().collect();
    let output_any: Tensor<bool> = run_graph(&graph_any, vec![("x", Tensor::new(&[4]).with_values(&[1, 2, 3, 4]).unwrap())]);
    assert!(output_any[0]);
    let x = Input::<i32>::new("x");
    let graph_all = x.sequence().filter(|v| v.gt_scalar(1)).map(|v| v.gt_scalar(0)).all().collect();
    let output_all: Tensor<bool> = run_graph(&graph_all, vec![("x", Tensor::new(&[4]).with_values(&[1, 2, 3, 4]).unwrap())]);
    assert!(output_all[0]);
}

#[test]
fn fold_composes_after_arbitrary_adapter_chain() {
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
fn reduce_boolean_composes_after_filter() {
    let x = Input::<bool>::new("x");
    let graph = x.sequence().filter(|v| v.eq_scalar(true)).reduce(|a, b| a.and(b)).collect();
    let output: Tensor<bool> = run_graph(&graph, vec![("x", Tensor::new(&[3]).with_values(&[true, true, false]).unwrap())]);
    assert!(!output[0]);
}
