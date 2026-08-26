use std::collections::HashSet;
use std::rc::Rc;

use crate::op::Op;

/// A materialized computation graph.
#[derive(Clone, Debug)]
pub struct Graph {
    operations: Vec<Rc<Op>>,
    output: Rc<Op>,
}

impl Graph {
    pub(crate) fn from_output(output: Rc<Op>) -> Self {
        let mut operations = Vec::new();
        let mut visited = HashSet::new();
        collect(&output, &mut visited, &mut operations);
        Self { operations, output }
    }

    pub fn operations(&self) -> &[Rc<Op>] { &self.operations }
    pub fn output(&self) -> &Rc<Op> { &self.output }

    /// Lowers this GraphPack graph into an executable TensorFlow graph.
    pub fn to_tensorflow(&self) -> Result<tensorflow::Graph, String> {
        crate::tensorflow::lower(self)
    }
}

fn collect(op: &Rc<Op>, visited: &mut HashSet<*const Op>, operations: &mut Vec<Rc<Op>>) {
    let ptr = Rc::as_ptr(op);
    if !visited.insert(ptr) { return; }
    for input in op.inputs() { collect(input, visited, operations); }
    operations.push(op.clone());
}
