pub use graphpack_macros::graphpack;

#[cfg(test)]
mod tests {
    use super::graphpack;

    #[test]
    fn graphpack_unit_closure_produces_graph_def() {
        let graph_def = graphpack!(|| {});

        assert!(!graph_def.is_empty());
    }
}
