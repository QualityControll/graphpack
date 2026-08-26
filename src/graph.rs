use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::op::{NodeId, Op};

thread_local! {
    static ARENA: RefCell<Vec<Rc<Op>>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn insert(op: Op) -> NodeId {
    ARENA.with(|arena| {
        let mut arena = arena.borrow_mut();
        let id = NodeId(arena.len());
        arena.push(Rc::new(op));
        id
    })
}

pub(crate) fn get(id: NodeId) -> Rc<Op> {
    ARENA.with(|arena| arena.borrow()[id.0].clone())
}

/// A materialized computation graph.
#[derive(Clone, Debug)]
pub struct Graph {
    operations: Vec<Rc<Op>>,
    output: NodeId,
}

impl Graph {
    pub(crate) fn from_output(output: NodeId) -> Self {
        let mut operations = Vec::new();
        let mut visited = HashSet::new();
        collect(output, &mut visited, &mut operations);
        Self { operations, output }
    }

    pub fn operations(&self) -> &[Rc<Op>] { &self.operations }
    pub fn output(&self) -> &Rc<Op> { &self.operations.iter().find(|op| std::ptr::eq(Rc::as_ptr(op), Rc::as_ptr(&get(self.output)))).unwrap() }
    pub fn to_tensorflow(&self) -> Result<tensorflow::Graph, String> { crate::tensorflow::lower(self) }
}

fn collect(id: NodeId, visited: &mut HashSet<NodeId>, operations: &mut Vec<Rc<Op>>) {
    if !visited.insert(id) { return; }
    let op = get(id);
    for &input in op.inputs() { collect(input, visited, operations); }
    operations.push(op);
}
