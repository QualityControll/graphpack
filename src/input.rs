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
                OpKind::Input { name: name.into(), dtype: T::scalar_type() },
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

/// Maps a tuple of inputs while borrowing each graph value for the duration
/// of the closure. This lets tuple elements be used more than once without
/// requiring callers to clone them explicitly.
pub trait InputTupleMap: Sized {
    type GraphValues<'a>
    where
        Self: 'a;

    fn map<U, Func>(self, f: Func) -> GraphValue<U>
    where
        Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U>;
}

macro_rules! impl_input_tuple_map {
    ($(($($input:ident, $value:ident),+)),+ $(,)?) => {
        $(
            impl<$($input: GraphType),+> InputTupleMap for ($(Input<$input>,)+) {
                type GraphValues<'a> = ($( &'a GraphValue<$input>, )+)
                where
                    Self: 'a;

                fn map<U, Func>(self, f: Func) -> GraphValue<U>
                where
                    Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U>,
                {
                    let ($( $value, )+) = self;
                    let values = ($( GraphValue::from_op($value.op), )+);
                    f(($( &values.$value, )+))
                }
            }
        )+
    };
}

// The macro above uses tuple element identifiers as field selectors only for
// documentation; the concrete implementations below use positional fields.
impl<A: GraphType, B: GraphType> InputTupleMap for (Input<A>, Input<B>) {
    type GraphValues<'a> = (&'a GraphValue<A>, &'a GraphValue<B>);

    fn map<U, Func>(self, f: Func) -> GraphValue<U>
    where
        Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U>,
    {
        let (a, b) = self;
        let values = (GraphValue::from_op(a.op), GraphValue::from_op(b.op));
        f((&values.0, &values.1))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType> InputTupleMap
    for (Input<A>, Input<B>, Input<C>)
{
    type GraphValues<'a> = (&'a GraphValue<A>, &'a GraphValue<B>, &'a GraphValue<C>);

    fn map<U, Func>(self, f: Func) -> GraphValue<U>
    where
        Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U>,
    {
        let (a, b, c) = self;
        let values = (
            GraphValue::from_op(a.op),
            GraphValue::from_op(b.op),
            GraphValue::from_op(c.op),
        );
        f((&values.0, &values.1, &values.2))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType> InputTupleMap
    for (Input<A>, Input<B>, Input<C>, Input<D>)
{
    type GraphValues<'a> = (
        &'a GraphValue<A>,
        &'a GraphValue<B>,
        &'a GraphValue<C>,
        &'a GraphValue<D>,
    );

    fn map<U, Func>(self, f: Func) -> GraphValue<U>
    where
        Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U>,
    {
        let (a, b, c, d) = self;
        let values = (
            GraphValue::from_op(a.op),
            GraphValue::from_op(b.op),
            GraphValue::from_op(c.op),
            GraphValue::from_op(d.op),
        );
        f((&values.0, &values.1, &values.2, &values.3))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType, E: GraphType> InputTupleMap
    for (Input<A>, Input<B>, Input<C>, Input<D>, Input<E>)
{
    type GraphValues<'a> = (
        &'a GraphValue<A>, &'a GraphValue<B>, &'a GraphValue<C>, &'a GraphValue<D>, &'a GraphValue<E>,
    );

    fn map<U, Func>(self, f: Func) -> GraphValue<U>
    where Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U> {
        let (a,b,c,d,e)=self;
        let values=(GraphValue::from_op(a.op),GraphValue::from_op(b.op),GraphValue::from_op(c.op),GraphValue::from_op(d.op),GraphValue::from_op(e.op));
        f((&values.0,&values.1,&values.2,&values.3,&values.4))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType, E: GraphType, F: GraphType> InputTupleMap
    for (Input<A>, Input<B>, Input<C>, Input<D>, Input<E>, Input<F>)
{
    type GraphValues<'a> = (&'a GraphValue<A>, &'a GraphValue<B>, &'a GraphValue<C>, &'a GraphValue<D>, &'a GraphValue<E>, &'a GraphValue<F>);
    fn map<U, Func>(self, f: Func) -> GraphValue<U> where Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U> {
        let (a,b,c,d,e,g)=self; let values=(GraphValue::from_op(a.op),GraphValue::from_op(b.op),GraphValue::from_op(c.op),GraphValue::from_op(d.op),GraphValue::from_op(e.op),GraphValue::from_op(g.op)); f((&values.0,&values.1,&values.2,&values.3,&values.4,&values.5))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType, E: GraphType, F: GraphType, G: GraphType> InputTupleMap
    for (Input<A>, Input<B>, Input<C>, Input<D>, Input<E>, Input<F>, Input<G>)
{
    type GraphValues<'a> = (&'a GraphValue<A>, &'a GraphValue<B>, &'a GraphValue<C>, &'a GraphValue<D>, &'a GraphValue<E>, &'a GraphValue<F>, &'a GraphValue<G>);
    fn map<U, Func>(self, f: Func) -> GraphValue<U> where Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U> {
        let (a,b,c,d,e,g,h)=self; let values=(GraphValue::from_op(a.op),GraphValue::from_op(b.op),GraphValue::from_op(c.op),GraphValue::from_op(d.op),GraphValue::from_op(e.op),GraphValue::from_op(g.op),GraphValue::from_op(h.op)); f((&values.0,&values.1,&values.2,&values.3,&values.4,&values.5,&values.6))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType, E: GraphType, F: GraphType, G: GraphType, H: GraphType> InputTupleMap
    for (Input<A>, Input<B>, Input<C>, Input<D>, Input<E>, Input<F>, Input<G>, Input<H>)
{
    type GraphValues<'a> = (&'a GraphValue<A>, &'a GraphValue<B>, &'a GraphValue<C>, &'a GraphValue<D>, &'a GraphValue<E>, &'a GraphValue<F>, &'a GraphValue<G>, &'a GraphValue<H>);
    fn map<U, Func>(self, f: Func) -> GraphValue<U> where Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U> {
        let (a,b,c,d,e,g,h,i)=self; let values=(GraphValue::from_op(a.op),GraphValue::from_op(b.op),GraphValue::from_op(c.op),GraphValue::from_op(d.op),GraphValue::from_op(e.op),GraphValue::from_op(g.op),GraphValue::from_op(h.op),GraphValue::from_op(i.op)); f((&values.0,&values.1,&values.2,&values.3,&values.4,&values.5,&values.6,&values.7))
    }
}

impl<A: GraphType, B: GraphType> InputTupleMap for ((Input<A>, Input<B>),) {
    type GraphValues<'a> = ((&'a GraphValue<A>, &'a GraphValue<B>),);
    fn map<U, Func>(self, f: Func) -> GraphValue<U> where Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U> {
        let ((a,b),)=self; let values=((GraphValue::from_op(a.op),GraphValue::from_op(b.op)),); f(((&values.0.0,&values.0.1),))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType> InputTupleMap for ((Input<A>, Input<B>), (Input<C>, Input<D>)) {
    type GraphValues<'a> = ((&'a GraphValue<A>, &'a GraphValue<B>), (&'a GraphValue<C>, &'a GraphValue<D>));
    fn map<U, Func>(self, f: Func) -> GraphValue<U> where Func: for<'a> FnOnce(Self::GraphValues<'a>) -> GraphValue<U> {
        let ((a,b),(c,d))=self; let values=((GraphValue::from_op(a.op),GraphValue::from_op(b.op)),(GraphValue::from_op(c.op),GraphValue::from_op(d.op))); f(((&values.0.0,&values.0.1),(&values.1.0,&values.1.1)))
    }
}

impl<A: GraphType,B: GraphType,C: GraphType,D: GraphType,E: GraphType,F: GraphType,G: GraphType,H: GraphType> InputTupleMap for ((Input<A>,Input<B>,Input<C>,Input<D>),(Input<E>,Input<F>,Input<G>,Input<H>)) {
    type GraphValues<'a> = ((&'a GraphValue<A>,&'a GraphValue<B>,&'a GraphValue<C>,&'a GraphValue<D>),(&'a GraphValue<E>,&'a GraphValue<F>,&'a GraphValue<G>,&'a GraphValue<H>));
    fn map<U, Func>(self,f:Func)->GraphValue<U> where Func:for<'a> FnOnce(Self::GraphValues<'a>)->GraphValue<U>{
        let ((a,b,c,d),(e,f,g,h))=self; let values=((GraphValue::from_op(a.op),GraphValue::from_op(b.op),GraphValue::from_op(c.op),GraphValue::from_op(d.op)),(GraphValue::from_op(e.op),GraphValue::from_op(f.op),GraphValue::from_op(g.op),GraphValue::from_op(h.op))); f(((&values.0.0,&values.0.1,&values.0.2,&values.0.3),(&values.1.0,&values.1.1,&values.1.2,&values.1.3)))
    }
}
