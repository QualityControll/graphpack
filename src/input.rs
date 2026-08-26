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
            op: Rc::new(Op::new(OpKind::Input { name: name.into(), dtype: T::scalar_type() }, Vec::new())),
            _marker: PhantomData,
        }
    }

    pub fn map<U, F>(self, f: F) -> U where F: FnOnce(&GraphValue<T>) -> U {
        let value = GraphValue::from_op(self.op);
        f(&value)
    }

    pub fn filter<F>(self, predicate: F) -> GraphValue<T> where F: FnOnce(GraphValue<T>) -> GraphValue<bool> {
        let value = GraphValue::from_op(self.op);
        let predicate = predicate(value.clone());
        GraphValue::from_op(Rc::new(Op::new(OpKind::Filter, vec![value.op().clone(), predicate.op().clone()])))
    }

    pub fn fold<U, F>(self, _init: U, _f: F) -> U where F: FnOnce(U, GraphValue<T>) -> U { todo!("fold graph construction is not implemented yet") }
    pub fn reduce<F>(self, _f: F) -> T where F: FnOnce(GraphValue<T>, GraphValue<T>) -> GraphValue<T> { todo!("reduce graph construction is not implemented yet") }
    pub fn scan<U, F>(self, _init: U, _f: F) -> Input<U> where F: FnOnce(U, GraphValue<T>) -> U { todo!("scan graph construction is not implemented yet") }

    pub fn collect(self) -> tensorflow::Graph {
        Graph::from_output(self.op).to_tensorflow().expect("failed to lower GraphPack graph to TensorFlow")
    }

    pub(crate) fn op(&self) -> &Rc<Op> { &self.op }
}

pub trait InputTupleMap {
    fn map<U, Func>(self, f: Func) -> U;
}

macro_rules! impl_input_tuple_map {
    ($(($($input:ident, $value:ident, $index:tt),+)),+ $(,)?) => {
        $(
            impl<$($input: GraphType),+> InputTupleMap for ($(Input<$input>,)+) {
                fn map<U, Func>(self, f: Func) -> U {
                    let ($( $value, )+) = self;
                    let values = ($( GraphValue::from_op($value.op), )+);
                    f(($( &values.$index, )+))
                }
            }
        )+
    };
}

impl_input_tuple_map!(
    (A, a, 0, B, b, 1),
    (A, a, 0, B, b, 1, C, c, 2),
    (A, a, 0, B, b, 1, C, c, 2, D, d, 3),
    (A, a, 0, B, b, 1, C, c, 2, D, d, 3, E, e, 4),
    (A, a, 0, B, b, 1, C, c, 2, D, d, 3, E, e, 4, F, f, 5),
    (A, a, 0, B, b, 1, C, c, 2, D, d, 3, E, e, 4, F, f, 5, G, g, 6),
    (A, a, 0, B, b, 1, C, c, 2, D, d, 3, E, e, 4, F, f, 5, G, g, 6, H, h, 7),
);

impl<A: GraphType, B: GraphType> InputTupleMap for ((Input<A>, Input<B>),) {
    fn map<U, Func>(self, f: Func) -> U {
        let ((a, b),) = self;
        let values = ((GraphValue::from_op(a.op), GraphValue::from_op(b.op)),);
        f(((&values.0.0, &values.0.1),))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType> InputTupleMap for ((Input<A>, Input<B>), (Input<C>, Input<D>)) {
    fn map<U, Func>(self, f: Func) -> U {
        let ((a, b), (c, d)) = self;
        let values = ((GraphValue::from_op(a.op), GraphValue::from_op(b.op)), (GraphValue::from_op(c.op), GraphValue::from_op(d.op)));
        f(((&values.0.0, &values.0.1), (&values.1.0, &values.1.1)))
    }
}

impl<A: GraphType, B: GraphType, C: GraphType, D: GraphType, E: GraphType, F: GraphType, G: GraphType, H: GraphType> InputTupleMap for ((Input<A>, Input<B>, Input<C>, Input<D>), (Input<E>, Input<F>, Input<G>, Input<H>)) {
    fn map<U, Func>(self, func: Func) -> U {
        let ((a, b, c, d), (e, f, g, h)) = self;
        let values = ((GraphValue::from_op(a.op), GraphValue::from_op(b.op), GraphValue::from_op(c.op), GraphValue::from_op(d.op)), (GraphValue::from_op(e.op), GraphValue::from_op(f.op), GraphValue::from_op(g.op), GraphValue::from_op(h.op)));
        func(((&values.0.0, &values.0.1, &values.0.2, &values.0.3), (&values.1.0, &values.1.1, &values.1.2, &values.1.3)))
    }
}
