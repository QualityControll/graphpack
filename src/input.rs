use crate::{GraphType, GraphValue};
use std::rc::Rc;

#[derive(Clone)]
pub struct Input<T: GraphType> {
    pub(crate) name: String,
    pub(crate) op: Rc<crate::op::Op>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: GraphType> Input<T> {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            op: Rc::new(crate::op::Op::new(crate::op::OpKind::Input {
                name: name.to_string(),
            })),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn map<U, F>(self, f: F) -> GraphValue<U>
    where
        U: GraphType,
        F: FnOnce(GraphValue<T>) -> GraphValue<U>,
    {
        let value = GraphValue::from_op(self.op);
        f(value)
    }

    pub fn filter<F>(self, predicate: F) -> Self
    where
        F: FnOnce(GraphValue<T>) -> GraphValue<bool>,
    {
        let value = GraphValue::from_op(self.op.clone());
        let predicate = predicate(value);
        Self {
            name: self.name,
            op: Rc::new(crate::op::Op::new(crate::op::OpKind::Filter {
                input: self.op,
                predicate: predicate.op,
            })),
            _marker: std::marker::PhantomData,
        }
    }
}

pub trait InputTupleMap {
    type GraphValues;

    fn map<U, Func>(self, f: Func) -> GraphValue<U>
    where
        U: GraphType,
        Func: FnOnce(Self::GraphValues) -> GraphValue<U>;
}

impl<A: GraphType, B: GraphType> InputTupleMap for (Input<A>, Input<B>) {
    type GraphValues = (GraphValue<A>, GraphValue<B>);

    fn map<U, Func>(self, f: Func) -> GraphValue<U>
    where
        U: GraphType,
        Func: FnOnce(Self::GraphValues) -> GraphValue<U>,
    {
        let (a, b) = self;
        f((GraphValue::from_op(a.op), GraphValue::from_op(b.op)))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType> InputTupleMap
    for (Input<A>, Input<B>, Input<C>)
{
    type GraphValues = (GraphValue<A>, GraphValue<B>, GraphValue<C>);

    fn map<U, Func>(self, f: Func) -> GraphValue<U>
    where
        U: GraphType,
        Func: FnOnce(Self::GraphValues) -> GraphValue<U>,
    {
        let (a, b, c) = self;
        f((
            GraphValue::from_op(a.op),
            GraphValue::from_op(b.op),
            GraphValue::from_op(c.op),
        ))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType> InputTupleMap
    for (Input<A>, Input<B>, Input<C>, Input<D>)
{
    type GraphValues = (
        GraphValue<A>,
        GraphValue<B>,
        GraphValue<C>,
        GraphValue<D>,
    );

    fn map<U, Func>(self, f: Func) -> GraphValue<U>
    where
        U: GraphType,
        Func: FnOnce(Self::GraphValues) -> GraphValue<U>,
    {
        let (a, b, c, d) = self;
        f((
            GraphValue::from_op(a.op),
            GraphValue::from_op(b.op),
            GraphValue::from_op(c.op),
            GraphValue::from_op(d.op),
        ))
    }
}

impl<
        A: GraphType,
        B: GraphType,
        C: GraphType,
        D: GraphType,
        E: GraphType,
        F: GraphType,
    > InputTupleMap
    for (
        (Input<A>, Input<B>, Input<C>),
        (Input<D>, Input<E>, Input<F>),
    )
{
    type GraphValues = (
        (GraphValue<A>, GraphValue<B>, GraphValue<C>),
        (GraphValue<D>, GraphValue<E>, GraphValue<F>),
    );

    fn map<U, Func>(self, func: Func) -> GraphValue<U>
    where
        U: GraphType,
        Func: FnOnce(Self::GraphValues) -> GraphValue<U>,
    {
        let ((a, b, c), (d, e, f)) = self;
        func((
            (
                GraphValue::from_op(a.op),
                GraphValue::from_op(b.op),
                GraphValue::from_op(c.op),
            ),
            (
                GraphValue::from_op(d.op),
                GraphValue::from_op(e.op),
                GraphValue::from_op(f.op),
            ),
        ))
    }
}

impl<
        A: GraphType,
        B: GraphType,
        C: GraphType,
        D: GraphType,
        E: GraphType,
        F: GraphType,
        G: GraphType,
        H: GraphType,
    > InputTupleMap
    for (
        (Input<A>, Input<B>, Input<C>, Input<D>),
        (Input<E>, Input<F>, Input<G>, Input<H>),
    )
{
    type GraphValues = (
        (GraphValue<A>, GraphValue<B>, GraphValue<C>, GraphValue<D>),
        (GraphValue<E>, GraphValue<F>, GraphValue<G>, GraphValue<H>),
    );

    fn map<U, Func>(self, func: Func) -> GraphValue<U>
    where
        U: GraphType,
        Func: FnOnce(Self::GraphValues) -> GraphValue<U>,
    {
        let ((a, b, c, d), (e, f, g, h)) = self;
        func((
            (
                GraphValue::from_op(a.op),
                GraphValue::from_op(b.op),
                GraphValue::from_op(c.op),
                GraphValue::from_op(d.op),
            ),
            (
                GraphValue::from_op(e.op),
                GraphValue::from_op(f.op),
                GraphValue::from_op(g.op),
                GraphValue::from_op(h.op),
            ),
        ))
    }
}
