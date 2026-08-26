use std::marker::PhantomData;
use std::rc::Rc;

/// A value in a GraphPack computation graph.
///
/// `Input<T>` is a typed handle to a graph value. It does not contain the
/// runtime data; it identifies a value produced by an [`Op`].
#[derive(Clone)]
pub struct Input<T> {
    op: Rc<Op>,
    _marker: PhantomData<T>,
}

/// A node in the GraphPack computation graph.
///
/// `Op` is intentionally small at this stage. It records the operation kind
/// and its input dependencies while leaving serialization and execution for a
/// later layer.
#[derive(Clone, Debug)]
pub struct Op {
    kind: OpKind,
    inputs: Vec<Rc<Op>>,
}

/// The kinds of graph nodes currently needed by the public graph model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpKind {
    Input { name: String },
}

impl<T> Input<T> {
    /// Creates a named graph input.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            op: Rc::new(Op {
                kind: OpKind::Input { name: name.into() },
                inputs: Vec::new(),
            }),
            _marker: PhantomData,
        }
    }

    /// Returns the operation represented by this value.
    pub(crate) fn op(&self) -> &Rc<Op> {
        &self.op
    }

    /// Applies a transformation to each element.
    pub fn map<U, F>(self, _f: F) -> Input<U>
    where
        F: FnOnce(T) -> U,
    {
        todo!("map graph construction is not implemented yet")
    }

    /// Retains elements matching a predicate.
    pub fn filter<F>(self, _predicate: F) -> Input<T>
    where
        F: FnOnce(&T) -> bool,
    {
        todo!("filter graph construction is not implemented yet")
    }

    /// Folds the input into a single value.
    pub fn fold<U, F>(self, _init: U, _f: F) -> U
    where
        F: FnOnce(U, T) -> U,
    {
        todo!("fold graph construction is not implemented yet")
    }

    /// Reduces the input into a single value.
    pub fn reduce<F>(self, _f: F) -> T
    where
        F: FnOnce(T, T) -> T,
    {
        todo!("reduce graph construction is not implemented yet")
    }

    /// Produces the intermediate accumulated values.
    pub fn scan<U, F>(self, _init: U, _f: F) -> Input<U>
    where
        F: FnOnce(U, T) -> U,
    {
        todo!("scan graph construction is not implemented yet")
    }

    /// Materializes the sequence as a graph value.
    pub fn collect(self) -> Self {
        self
    }
}

impl Op {
    /// Returns the kind of this graph operation.
    pub fn kind(&self) -> &OpKind {
        &self.kind
    }

    /// Returns the operation dependencies.
    pub fn inputs(&self) -> &[Rc<Op>] {
        &self.inputs
    }
}
