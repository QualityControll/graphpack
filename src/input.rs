use std::marker::PhantomData;

/// A value in a GraphPack computation graph.
///
/// `Input<T>` represents a graph input whose elements have type `T`. The
/// iterator-style methods are intentionally skeletal for now; their graph
/// construction semantics will be implemented incrementally.
pub struct Input<T> {
    _marker: PhantomData<T>,
}

impl<T> Input<T> {
    /// Creates a named graph input.
    pub fn new(_name: impl Into<String>) -> Self {
        Self {
            _marker: PhantomData,
        }
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
