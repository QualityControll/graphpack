use std::rc::Rc;

/// A node in the GraphPack computation graph.
#[derive(Clone, Debug)]
pub struct Op {
    kind: OpKind,
    inputs: Vec<Rc<Op>>,
}

/// The kind of computation represented by an operation node.
#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    Input { name: String },
    Constant { value: ConstantValue },
    Map,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConstantValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    Bool(bool),
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
