use std::marker::PhantomData;
use std::rc::Rc;

/// A typed value in a GraphPack computation graph.
#[derive(Clone)]
pub struct Input<T> {
    op: Rc<Op>,
    _marker: PhantomData<T>,
}

/// A node in the GraphPack computation graph.
#[derive(Clone, Debug)]
pub struct Op {
    kind: OpKind,
    inputs: Vec<Rc<Op>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpKind {
    Input { name: String },
    Map,
}

/// A typed graph value passed to a transformation closure.
#[derive(Clone)]
pub struct GraphValue<T> {
    op: Rc<Op>,
    _marker: PhantomData<T>,
}

impl<T> Input<T> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            op: Rc::new(Op {
                kind: OpKind::Input { name: name.into() },
                inputs: Vec::new(),
            }),
            _marker: PhantomData,
        }
    }

    /// Experimental `map` design: the closure receives a graph value rather
    /// than runtime data, allowing normal Rust expressions to build graphs.
    pub fn map<U, F>(self, f: F) -> Input<U>
    where
        F: FnOnce(GraphValue<T>) -> GraphValue<U>,
    {
        let value = GraphValue {
            op: self.op.clone(),
            _marker: PhantomData,
        };
        let result = f(value);
        Input {
            op: Rc::new(Op {
                kind: OpKind::Map,
                inputs: vec![self.op, result.op],
            }),
            _marker: PhantomData,
        }
    }

    pub(crate) fn op(&self) -> &Rc<Op> {
        &self.op
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

    pub fn collect(self) -> Self {
        self
    }
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

impl Op {
    pub fn kind(&self) -> &OpKind {
        &self.kind
    }

    pub fn inputs(&self) -> &[Rc<Op>] {
        &self.inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_accepts_graph_value_closure() {
        let input = Input::<f32>::new("x");
        let input_op = input.op().clone();
        let mapped = input.map(GraphValue::from_op);

        assert_eq!(mapped.op().kind(), &OpKind::Map);
        assert_eq!(mapped.op().inputs().len(), 2);
        assert!(Rc::ptr_eq(&mapped.op().inputs()[0], &input_op));
        assert!(Rc::ptr_eq(&mapped.op().inputs()[1], &input_op));
    }
}
