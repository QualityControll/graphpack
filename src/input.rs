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

    pub fn map<U, F>(self, f: F) -> U
    where
        F: FnOnce(&GraphValue<T>) -> U,
    {
        let value = GraphValue::from_op(self.op);
        f(&value)
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

pub trait InputTupleMap: Sized {
    type GraphValues<'a>
    where
        Self: 'a;

    fn map<U, Func>(&self, f: Func) -> U
    where
        for<'a> Func: FnOnce(Self::GraphValues<'a>) -> U;
}

macro_rules! impl_input_tuple_map {
    ($(($($input:ident, $value:ident),+)),+ $(,)?) => {
        $(
            impl<$($input: GraphType),+> InputTupleMap for ($(Input<$input>,)+) {
                type GraphValues<'a> = ($( &'a GraphValue<$input>, )+)
                where
                    Self: 'a;

                fn map<U, Func>(&self, f: Func) -> U
                where
                    for<'a> Func: FnOnce(Self::GraphValues<'a>) -> U,
                {
                    let ($( $value, )+) = self;
                    let values = ($(GraphValue::from_op($value.op().clone()),)+);
                    f(($( &values.$value, )+))
                }
            }
        )+
    };
}

impl_input_tuple_map!(
    (A, a, B, b),
    (A, a, B, b, C, c),
    (A, a, B, b, C, c, D, d),
    (A, a, B, b, C, c, D, d, E, e),
    (A, a, B, b, C, c, D, d, E, e, F, f),
    (A, a, B, b, C, c, D, d, E, e, F, f, G, g),
    (A, a, B, b, C, c, D, d, E, e, F, f, G, g, H, h),
);

impl<A: GraphType, B: GraphType> InputTupleMap for ((Input<A>, Input<B>),) {
    type GraphValues<'a> = ((&'a GraphValue<A>, &'a GraphValue<B>),)
    where
        Self: 'a;

    fn map<U, Func>(&self, f: Func) -> U
    where
        for<'a> Func: FnOnce(Self::GraphValues<'a>) -> U,
    {
        let ((a, b),) = self;
        let values = ((GraphValue::from_op(a.op().clone()), GraphValue::from_op(b.op().clone())),);
        f(((&values.0.0, &values.0.1),))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType>
    InputTupleMap for ((Input<A>, Input<B>), (Input<C>, Input<D>))
{
    type GraphValues<'a> = (
        (&'a GraphValue<A>, &'a GraphValue<B>),
        (&'a GraphValue<C>, &'a GraphValue<D>),
    )
    where
        Self: 'a;

    fn map<U, Func>(&self, f: Func) -> U
    where
        for<'a> Func: FnOnce(Self::GraphValues<'a>) -> U,
    {
        let ((a, b), (c, d)) = self;
        let values = (
            (GraphValue::from_op(a.op().clone()), GraphValue::from_op(b.op().clone())),
            (GraphValue::from_op(c.op().clone()), GraphValue::from_op(d.op().clone())),
        );
        f(((&values.0.0, &values.0.1), (&values.1.0, &values.1.1)))
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
    type GraphValues<'a> = (
        (
            &'a GraphValue<A>,
            &'a GraphValue<B>,
            &'a GraphValue<C>,
            &'a GraphValue<D>,
        ),
        (
            &'a GraphValue<E>,
            &'a GraphValue<F>,
            &'a GraphValue<G>,
            &'a GraphValue<H>,
        ),
    )
    where
        Self: 'a;

    fn map<U, Func>(&self, f: Func) -> U
    where
        for<'a> Func: FnOnce(Self::GraphValues<'a>) -> U,
    {
        let ((a, b, c, d), (e, f, g, h)) = self;
        let values = (
            (
                GraphValue::from_op(a.op().clone()),
                GraphValue::from_op(b.op().clone()),
                GraphValue::from_op(c.op().clone()),
                GraphValue::from_op(d.op().clone()),
            ),
            (
                GraphValue::from_op(e.op().clone()),
                GraphValue::from_op(f.op().clone()),
                GraphValue::from_op(g.op().clone()),
                GraphValue::from_op(h.op().clone()),
            ),
        );
        f((
            (&values.0.0, &values.0.1, &values.0.2, &values.0.3),
            (&values.1.0, &values.1.1, &values.1.2, &values.1.3),
        ))
    }
}
