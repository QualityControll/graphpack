#[cfg(test)]
mod tests {
    use super::{graphpack, run_graph, run_graph_i32, Input};

    #[test]
    fn graphpack_multi_line_arithmetic_runs() {
        let graph_def = graphpack!(|x: Input<f32>| {
            let a = x + 1.0;
            let b = a * 2.0;
            b - 3.0
        });

        let output = run_graph(&graph_def, &[("x", &[1.0, 2.0, 3.0])], "output").unwrap();
        assert_eq!(output.as_ref(), &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn graphpack_bitwise_runs() {
        let graph_def = graphpack!(|x: Input<i32>| {
            let a = x & 0xff;
            let b = a << 1;
            b | 1
        });

        let output = run_graph_i32(&graph_def, &[("x", &[1, 2, 3])], "output").unwrap();
        assert_eq!(output.as_ref(), &[3, 5, 7]);
    }

    #[test]
    fn graphpack_unary_arithmetic_runs() {
        let graph_def = graphpack!(|x: Input<f32>| {
            let y = -x;
            y + 2.0
        });

        let output = run_graph(&graph_def, &[("x", &[1.0, 2.0, 3.0])], "output").unwrap();
        assert_eq!(output.as_ref(), &[1.0, 0.0, -1.0]);
    }
}