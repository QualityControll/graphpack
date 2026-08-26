use std::marker::PhantomData;

pub use graphpack_macros::graphpack;

pub struct Input<T> {
    _marker: PhantomData<T>,
}

#[cfg(test)]
mod tests {
    use super::{graphpack, Input};

    #[test]
    fn graphpack_unit_closure_produces_graph_def() {
        let graph_def = graphpack!(|| {});

        assert!(!graph_def.is_empty());
    }

    #[test]
    fn graphpack_input_produces_placeholder() {
        let graph_def = graphpack!(|_x: Input<f32>| {});

        assert!(!graph_def.is_empty());
    }
}
