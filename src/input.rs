use std::marker::PhantomData;
use std::rc::Rc;

use crate::graph::Graph;
use crate::graph_value::GraphValue;
use crate::op::{Op, OpKind};

/// A typed value entering a GraphPack computation graph.
#[derive(Clone)]
pub struct Input<T> {
    op: Rc<Op>,
    _marker: PhantomData<T>,
}

impl<T> Input<T> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            op: Rc::new(Op::new(OpKind::Input { name: name.into() }, Vec::new())),
            _marker: PhantomData,
        }
    }

    /// Applies a graph-building function to each value produced by this input.
    pub fn map<U, F>(self, f: F) -> GraphValue<U>
    where
        F: FnOnce(GraphValue<T>) -> GraphValue<U>,
    {
        f(GraphValue::from_op(self.op))
    }

    pub fn filter<F>(self, _predicate: F) -> Input<T>
    where
        F: FnOnce(&GraphValue<T>) -> GraphValue<bool>,
    {
        todo!("filter graph construction is not implemented yet")
    }

    pub fn fold<U, F>(self, _init: U, _f: F) -> U
    where
        F: FnOnce(U, GraphValue<T>) -> U,
    {
        todo!("fold graph construction is not implemented yet")
    }

    pub fn reduce<F>(self, _f: F) -> T
    where
        F: FnOnce(GraphValue<T>, GraphValue<T>) -> GraphValue<T>,
    {
        todo!("reduce graph construction is not implemented yet")
    }

    pub fn scan<U, F>(self, _init: U, _f: F) -> Input<U>
    where
        F: FnOnce(U, GraphValue<T>) -> U,
    {
        todo!("scan graph construction is not implemented yet")
    }

    /// Completes the lazy pipeline and returns its concrete graph.
    pub fn collect(self) -> Graph {
        Graph::from_output(self.op)
    }

    pub(crate) fn op(&self) -> &Rc<Op> {
        &self.op
    }
}
