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
    let graph = x.sequence().filter(|v| v).reduce(|a, b| a.and(b)).collect();
    let output: Tensor<bool> = run_graph(&graph, vec![("x", Tensor::new(&[3]).with_values(&[true, true, false]).unwrap())]);
    assert!(!output[0]);
}