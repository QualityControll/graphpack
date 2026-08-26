/// A TensorFlow GraphDef serialized as protobuf bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphDef {
    bytes: Vec<u8>,
}

impl GraphDef {
    /// Creates an empty GraphDef.
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Returns the serialized GraphDef protobuf.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns whether this GraphDef contains no serialized data.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl Default for GraphDef {
    fn default() -> Self {
        Self::new()
    }
}

pub use graphpack_macros::graphpack;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphpack_unit_closure_produces_empty_graph_def() {
        let graph = graphpack!(|| {});

        assert!(graph.is_empty());
        assert_eq!(graph.as_bytes(), &[]);
    }
}
