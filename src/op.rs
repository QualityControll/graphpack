use std::rc::Rc;

/// A node in the GraphPack computation graph.
#[derive(Clone, Debug)]
pub struct Op {
    kind: OpKind,
    inputs: Vec<Rc<Op>>,
}

/// The kind of computation represented by an operation node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpKind {
    Input { name: String },
    Map,
}

impl Op {
    pub(crate) fn new(kind: OpKind, inputs: Vec<Rc<Op>>) -> Self {
        Self { kind, inputs }
    }

    pub fn kind(&self) -> &OpKind {
        &self.kind
    }

    pub fn inputs(&self) -> &[Rc<Op>] {
        &self.inputs
    }
}
