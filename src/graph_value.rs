use std::marker::PhantomData;
use std::rc::Rc;

use crate::op::Op;

/// A typed value flowing through a GraphPack computation graph.
///
/// `GraphValue<T>` contains no runtime `T`; it is a typed handle to an
/// operation in the graph.
#[derive(Clone)]
pub struct GraphValue<T> {
    pub(crate) op: Rc<Op>,
    pub(crate) _marker: PhantomData<T>,
}

impl<T> GraphValue<T> {
    pub(crate) fn from_op(op: Rc<Op>) -> Self {
        Self {
            op,
            _marker: PhantomData,
        }
    }

    pub(crate) fn op(&self) -> &Rc<Op> {
        &self.op
    }
}
