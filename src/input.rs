use std::marker::PhantomData;
use std::rc::Rc;

use crate::graph::Graph;
use crate::graph_value::GraphValue;
use crate::op::{GraphType, Op, OpKind};

#[derive(Clone)]
pub struct Input<T> {
    op: Rc<Op>,
    _marker: PhantomData<T>,
}

impl<T: GraphType> Input<T> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            op: Rc::new(Op::new(
                OpKind::Input {
                    name: name.into(),
                    dtype: T::scalar_type(),
                },
                Vec::new(),
            )),
            _marker: PhantomData,
        }
    }

    pub fn map<U, F>(self, f: F) -> GraphValue<U>
    where
        F: FnOnce(GraphValue<T>) -> GraphValue<U>,
    {
        f(GraphValue::from_op(self.op))
    }

    pub fn filter<F>(self, predicate: F) -> GraphValue<T>
    where
        F: FnOnce(GraphValue<T>) -> GraphValue<bool>,
    {
        let value = GraphValue::from_op(self.op);
        let predicate = predicate(value.clone());
        GraphValue::from_op(Rc::new(Op::new(
            OpKind::Filter,
            vec![value.op().clone(), predicate.op().clone()],
        )))
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

    pub fn collect(self) -> tensorflow::Graph {
        Graph::from_output(self.op)
            .to_tensorflow()
            .expect("failed to lower GraphPack graph to TensorFlow")
    }

    pub(crate) fn op(&self) -> &Rc<Op> {
        &self.op
    }
}